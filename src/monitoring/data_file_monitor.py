"""
Auto-ingestion system for data/ directory files
Monitors file changes and automatically triggers ingestion to keep database in sync
"""
import os
import asyncio
import hashlib
import logging
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config.settings import settings
from src.data_layer.database.models import (
    DataFileIngestion,
    FileIngestionStatus,
    Base
)

logger = logging.getLogger(__name__)


class DataFileEventHandler(FileSystemEventHandler):

    def __init__(self, monitor: 'DataFileMonitor'):
        self.monitor = monitor
        self.debounce_map: Dict[str, float] = {}  # file_path -> last_event_time
        self.debounce_seconds = 2.0  # Wait 2 seconds before processing

    def _should_process(self, file_path: str) -> bool:
        if file_path.startswith('.') or file_path.endswith(('~', '.tmp', '.swp')):
            return False

        valid_extensions = {'.md', '.csv', '.yaml', '.yml', '.json'}
        if not any(file_path.endswith(ext) for ext in valid_extensions):
            return False

        current_time = time.time()
        last_event_time = self.debounce_map.get(file_path, 0)

        if current_time - last_event_time < self.debounce_seconds:
            return False

        self.debounce_map[file_path] = current_time
        return True

    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return

        file_path = event.src_path
        if self._should_process(file_path):
            logger.info(f"File created: {file_path}")
            self.monitor.schedule_ingestion(file_path, event_type="created")

    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return

        file_path = event.src_path
        if self._should_process(file_path):
            logger.info(f"File modified: {file_path}")
            self.monitor.schedule_ingestion(file_path, event_type="modified")

    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory:
            return

        file_path = event.src_path
        if self._should_process(file_path):
            logger.info(f"File deleted: {file_path}")
            self.monitor.mark_file_deleted(file_path)


class DataFileMonitor:

    def __init__(self):
        self.data_dir = Path(settings.PROJECT_ROOT) / "data"
        self.engine = create_engine(settings.DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)

        self.observer: Optional[Observer] = None
        self.event_handler: Optional[DataFileEventHandler] = None
        self.ingestion_queue: List[Dict] = []
        self.ingestion_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self._vector_store: Optional['PgVectorStore'] = None

        logger.info(f"Initialized DataFileMonitor for directory: {self.data_dir}")

    def _get_vector_store(self):
        if self._vector_store is None:
            from src.data_layer.vector_store.pgvector_store import PgVectorStore
            self._vector_store = PgVectorStore()
            logger.info("Initialized shared PgVectorStore instance")
        return self._vector_store

    def start(self):
        logger.info("Starting DataFileMonitor...")

        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_table_exists()

        self._initial_scan()

        self.event_handler = DataFileEventHandler(self)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, str(self.data_dir), recursive=True)
        self.observer.start()

        self.ingestion_thread = threading.Thread(target=self._ingestion_worker, daemon=True)
        self.ingestion_thread.start()

        logger.info("DataFileMonitor started successfully")

    def stop(self):
        logger.info("Stopping DataFileMonitor...")

        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)

        self.stop_event.set()
        if self.ingestion_thread:
            self.ingestion_thread.join(timeout=5)

        logger.info("DataFileMonitor stopped")

    def _ensure_table_exists(self):
        try:
            Base.metadata.create_all(self.engine, tables=[DataFileIngestion.__table__])
            logger.info("data_file_ingestion table ready")
        except Exception as e:
            logger.error(f"Error creating table: {e}")

    def _initial_scan(self):
        logger.info("Performing initial scan of data/ directory...")

        db = self.SessionLocal()
        try:
            tracked_files = {
                record.file_path: record
                for record in db.query(DataFileIngestion).all()
            }

            scanned_files: Set[str] = set()
            for file_path in self._scan_directory(self.data_dir):
                rel_path = self._get_relative_path(file_path)
                scanned_files.add(rel_path)

                if rel_path in tracked_files:
                    record = tracked_files[rel_path]
                    file_hash = self._calculate_file_hash(file_path)

                    if file_hash != record.file_hash:
                        logger.info(f"Detected modified file: {rel_path}")
                        self.schedule_ingestion(str(file_path), event_type="modified")
                else:
                    logger.info(f"Detected new file: {rel_path}")
                    self.schedule_ingestion(str(file_path), event_type="created")

            for rel_path, record in tracked_files.items():
                if rel_path not in scanned_files and record.status != FileIngestionStatus.DELETED:
                    logger.info(f"File no longer exists: {rel_path}")
                    record.status = FileIngestionStatus.DELETED
                    record.updated_at = datetime.utcnow()

            db.commit()
            logger.info(f"Initial scan complete. Found {len(scanned_files)} files")

            pending_records = db.query(DataFileIngestion).filter(
                DataFileIngestion.status == FileIngestionStatus.PENDING
            ).all()

            if pending_records:
                logger.info(f"Found {len(pending_records)} PENDING records in database - queuing for processing")
                for record in pending_records:
                    full_path = Path(settings.PROJECT_ROOT) / record.file_path
                    if full_path.exists():
                        logger.info(f"Queuing PENDING record: {record.file_path}")
                        self.schedule_ingestion(str(full_path), event_type="pending_retry")
                    else:
                        logger.warning(f"PENDING record points to non-existent file: {record.file_path}")
                        record.status = FileIngestionStatus.DELETED
                        record.updated_at = datetime.utcnow()
                db.commit()

        except Exception as e:
            logger.error(f"Error during initial scan: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()

    def _scan_directory(self, directory: Path) -> List[Path]:
        valid_extensions = {'.md', '.csv', '.yaml', '.yml', '.json'}
        files = []

        for item in directory.rglob('*'):
            if item.is_file() and item.suffix in valid_extensions:
                if not any(part.startswith('.') for part in item.parts):
                    files.append(item)

        return files

    def _get_relative_path(self, file_path: Path) -> str:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        try:
            return str(file_path.relative_to(settings.PROJECT_ROOT))
        except ValueError:
            return str(file_path)

    def _calculate_file_hash(self, file_path: Path) -> str:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def _categorize_file(self, file_path: str) -> str:
        rel_path = file_path.replace(str(settings.PROJECT_ROOT), '').strip(os.sep)

        if 'knowledge_base' in rel_path:
            return 'knowledge_base'
        elif 'claim_library' in rel_path:
            return 'claim_library'
        elif 'competitors' in rel_path:
            return 'competitors'
        elif 'historical' in rel_path:
            return 'historical'
        elif 'governance' in rel_path:
            return 'governance'
        elif 'products' in rel_path and 'catalog' in rel_path.lower():
            return 'product_catalog'
        elif 'company' in rel_path and 'brand_voice' in rel_path.lower():
            return 'brand_voice'
        else:
            return 'other'

    def schedule_ingestion(self, file_path: str, event_type: str = "created"):
        self.ingestion_queue.append({
            'file_path': file_path,
            'event_type': event_type,
            'queued_at': time.time()
        })
        logger.debug(f"Scheduled ingestion for {file_path} (event: {event_type})")

    def mark_file_deleted(self, file_path: str):
        db = self.SessionLocal()
        try:
            rel_path = self._get_relative_path(Path(file_path))

            record = db.query(DataFileIngestion).filter_by(file_path=rel_path).first()
            if record:
                record.status = FileIngestionStatus.DELETED
                record.updated_at = datetime.utcnow()
                db.commit()
                logger.info(f"Marked file as deleted: {rel_path}")

        except Exception as e:
            logger.error(f"Error marking file as deleted: {e}")
            db.rollback()
        finally:
            db.close()

    def _check_pending_records(self):
        db = self.SessionLocal()
        try:
            pending_records = db.query(DataFileIngestion).filter(
                DataFileIngestion.status == FileIngestionStatus.PENDING
            ).limit(10).all()  # Process max 10 at a time

            if pending_records:
                logger.info(f"Found {len(pending_records)} PENDING records - queuing for processing")
                for record in pending_records:
                    full_path = Path(settings.PROJECT_ROOT) / record.file_path
                    if full_path.exists():
                        self.schedule_ingestion(str(full_path), event_type="pending_retry")
                    else:
                        logger.warning(f"PENDING record points to non-existent file: {record.file_path}")
                        record.status = FileIngestionStatus.DELETED
                        record.updated_at = datetime.utcnow()
                db.commit()

        except Exception as e:
            logger.error(f"Error checking PENDING records: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()

    def _ingestion_worker(self):
        logger.info("Ingestion worker started")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._worker_loop = loop

        last_pending_check = 0
        PENDING_CHECK_INTERVAL = 30  # Check for PENDING records every 30 seconds

        try:
            while not self.stop_event.is_set():
                try:
                    if self.ingestion_queue:
                        item = self.ingestion_queue.pop(0)
                        self._process_file(item['file_path'], item['event_type'])
                    else:
                        current_time = time.time()
                        if current_time - last_pending_check > PENDING_CHECK_INTERVAL:
                            self._check_pending_records()
                            last_pending_check = current_time

                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Error in ingestion worker: {e}", exc_info=True)
                    time.sleep(5)  # Back off on error
        finally:
            loop.close()
            logger.info("Ingestion worker stopped")

    def _process_file(self, file_path: str, event_type: str):
        db = self.SessionLocal()
        try:
            file_path_obj = Path(file_path)
            rel_path = self._get_relative_path(file_path_obj)

            if not file_path_obj.exists():
                logger.warning(f"File no longer exists: {rel_path}")
                return

            record = db.query(DataFileIngestion).filter_by(file_path=rel_path).first()

            if not record:
                record = DataFileIngestion(
                    file_path=rel_path,
                    file_name=file_path_obj.name,
                    file_type=file_path_obj.suffix,
                    file_category=self._categorize_file(file_path),
                    status=FileIngestionStatus.PENDING
                )
                db.add(record)

            record.file_size_bytes = file_path_obj.stat().st_size
            record.file_hash = self._calculate_file_hash(file_path_obj)
            record.last_modified_at = datetime.fromtimestamp(file_path_obj.stat().st_mtime)
            record.status = FileIngestionStatus.INGESTING
            record.ingestion_started_at = datetime.utcnow()

            db.commit()

            start_time = time.time()
            success = self._ingest_file_content(file_path_obj, record, db)
            duration = time.time() - start_time

            if success:
                record.status = FileIngestionStatus.INGESTED
                record.ingestion_completed_at = datetime.utcnow()
                record.ingestion_duration_seconds = duration
                record.error_message = None
                logger.info(f"Successfully ingested {rel_path} in {duration:.2f}s")
            else:
                record.status = FileIngestionStatus.FAILED
                record.retry_count += 1
                logger.warning(f"Failed to ingest {rel_path}")

            db.commit()

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            if record:
                record.status = FileIngestionStatus.FAILED
                record.error_message = str(e)
                record.retry_count += 1
                db.commit()
        finally:
            db.close()

    def _ingest_file_content(self, file_path: Path, record: DataFileIngestion, db: Session) -> bool:
        try:
            category = record.file_category

            if category == 'knowledge_base':
                return self._ingest_knowledge_base_file(file_path, record, db)
            elif category == 'claim_library':
                return self._ingest_claim_library_file(file_path, record, db)
            elif category == 'competitors':
                return self._ingest_competitors_file(file_path, record, db)
            elif category == 'historical':
                return self._ingest_historical_file(file_path, record, db)
            elif category == 'product_catalog':
                return self._ingest_product_catalog_file(file_path, record, db)
            elif category == 'brand_voice':
                # Brand voice is loaded directly by ContentGenerator, no ingestion needed
                logger.info(f"Brand voice file detected: {file_path} - loaded directly by agents")
                return True
            else:
                logger.info(f"No ingestion logic for category '{category}' - marked as ingested")
                return True

        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}", exc_info=True)
            record.error_message = str(e)
            return False

    def _ingest_knowledge_base_file(self, file_path: Path, record: DataFileIngestion, db: Session) -> bool:
        try:
            # Use shared vector store instance to avoid reinitializing SentenceTransformer
            vector_store = self._get_vector_store()

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'category': record.file_category
            }

            if content.startswith('---'):
                try:
                    import frontmatter
                    from datetime import date, datetime
                    post = frontmatter.loads(content)

                    clean_metadata = {}
                    for key, value in post.metadata.items():
                        if isinstance(value, (date, datetime)):
                            clean_metadata[key] = value.isoformat()
                        elif isinstance(value, list):
                            clean_metadata[key] = [
                                v.isoformat() if isinstance(v, (date, datetime)) else v
                                for v in value
                            ]
                        else:
                            clean_metadata[key] = value

                    metadata.update(clean_metadata)
                    content = post.content
                except Exception as e:
                    logger.warning(f"Could not parse frontmatter: {e}")

            embedding = vector_store.embed_text(content)
            embedding_list = embedding.tolist()

            import psycopg2
            import json
            from src.config.settings import settings

            sync_db_url = settings.DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://')

            conn = psycopg2.connect(sync_db_url)
            try:
                cur = conn.cursor()

                embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
                metadata_str = json.dumps(metadata)

                query = f"""
                    INSERT INTO {vector_store.collection_name}
                    (content, embedding, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """

                cur.execute(query, (content, embedding_str, metadata_str))
                doc_id = cur.fetchone()[0]

                conn.commit()
                cur.close()

                doc_ids = [doc_id]
                logger.debug(f"Inserted document {doc_id} using sync connection")

            finally:
                conn.close()

            record.document_ids = doc_ids
            record.num_chunks = len(doc_ids)

            logger.info(f"Ingested {file_path.name} -> {len(doc_ids)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error ingesting knowledge base file: {e}", exc_info=True)
            return False

    def _ingest_claim_library_file(self, file_path: Path, record: DataFileIngestion, db: Session) -> bool:
        logger.info(f"Claim library file {file_path.name} tracked (loaded on demand by agents)")
        return True

    def _ingest_competitors_file(self, file_path: Path, record: DataFileIngestion, db: Session) -> bool:
        logger.info(f"Competitors file {file_path.name} tracked (loaded on demand by agents)")
        return True

    def _ingest_historical_file(self, file_path: Path, record: DataFileIngestion, db: Session) -> bool:
        logger.info(f"Historical data file {file_path.name} tracked (used by calibration)")
        return True

    def get_ingestion_status(self, limit: int = 100) -> List[Dict]:
        db = self.SessionLocal()
        try:
            records = db.query(DataFileIngestion).order_by(
                DataFileIngestion.updated_at.desc()
            ).limit(limit).all()

            return [
                {
                    'file_path': r.file_path,
                    'file_name': r.file_name,
                    'category': r.file_category,
                    'status': r.status.value,
                    'file_size': r.file_size_bytes,
                    'last_modified': r.last_modified_at.isoformat() if r.last_modified_at else None,
                    'ingested_at': r.ingestion_completed_at.isoformat() if r.ingestion_completed_at else None,
                    'duration': r.ingestion_duration_seconds,
                    'num_chunks': r.num_chunks,
                    'error': r.error_message
                }
                for r in records
            ]
        finally:
            db.close()

    def _ingest_product_catalog_file(self, file_path: Path, record: DataFileIngestion, db: Session) -> bool:
        """
        Ingest product catalog JSON into RAG vector store

        Per requirements Section 5:
        - Parse catalog.json modules and features
        - Create separate vector documents for each feature/use case
        - Store with embeddings for RAG retrieval
        """
        try:
            from src.data_layer.vector_store.pgvector_store import PgVectorStore
            import json
            import asyncio

            with open(file_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)

            logger.info(f"Ingesting product catalog: {file_path}")

            vector_store = PgVectorStore(collection_name="documents")

            document_ids = []
            chunk_count = 0

            product_name = catalog.get('product', 'Agentic AI')
            overview = catalog.get('overview', {})
            value_prop = overview.get('value_prop', '')

            def add_document_sync(content: str, metadata: dict) -> int:
                """
                Add document to vector store using synchronous database operations
                to avoid event loop conflicts in worker thread
                """
                import psycopg2
                from src.config.settings import settings

                embedding = vector_store.embed_text(content)
                embedding_list = embedding.tolist()

                sync_db_url = settings.DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://')

                conn = psycopg2.connect(sync_db_url)
                try:
                    cur = conn.cursor()

                    embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
                    metadata_str = json.dumps(metadata)

                    query = f"""
                        INSERT INTO {vector_store.collection_name}
                        (content, embedding, metadata)
                        VALUES (%s, %s, %s)
                        RETURNING id
                    """

                    cur.execute(query, (content, embedding_str, metadata_str))
                    doc_id = cur.fetchone()[0]

                    conn.commit()
                    cur.close()

                    return doc_id
                finally:
                    conn.close()

            if value_prop:
                doc_id = add_document_sync(
                    content=f"Product: {product_name}\n\nValue Proposition: {value_prop}",
                    metadata={
                        'source': str(file_path),
                        'type': 'product_overview',
                        'product': product_name
                    }
                )
                if doc_id:
                    document_ids.append(doc_id)
                    chunk_count += 1

            modules = catalog.get('modules', [])
            for module in modules:
                module_id = module.get('id', '')
                module_name = module.get('name', '')
                module_desc = module.get('description', '')
                features = module.get('features', [])

                module_content = f"""Module: {module_name}
ID: {module_id}
Description: {module_desc}

Features:
{chr(10).join('- ' + f for f in features)}

Inputs: {', '.join(module.get('inputs', []))}
Outputs: {', '.join(module.get('outputs', []))}
"""

                doc_id = add_document_sync(
                    content=module_content,
                    metadata={
                        'source': str(file_path),
                        'type': 'product_module',
                        'module_id': module_id,
                        'module_name': module_name,
                        'product': product_name
                    }
                )
                if doc_id:
                    document_ids.append(doc_id)
                    chunk_count += 1

                for idx, feature in enumerate(features):
                    feature_content = f"""Feature: {feature}
Module: {module_name} ({module_id})
Product: {product_name}

This feature is part of the {module_name} module, which {module_desc.lower()}
"""

                    doc_id = add_document_sync(
                        content=feature_content,
                        metadata={
                            'source': str(file_path),
                            'type': 'product_feature',
                            'module_id': module_id,
                            'module_name': module_name,
                            'feature': feature,
                            'product': product_name
                        }
                    )
                    if doc_id:
                        document_ids.append(doc_id)
                        chunk_count += 1

            workflows = catalog.get('workflows', [])
            for workflow in workflows:
                workflow_id = workflow.get('id', '')
                workflow_name = workflow.get('name', '')
                workflow_purpose = workflow.get('purpose', '')
                uses_modules = workflow.get('uses_modules', [])

                workflow_content = f"""Workflow: {workflow_name}
ID: {workflow_id}
Purpose: {workflow_purpose}

Uses Modules: {', '.join(uses_modules)}
Product: {product_name}
"""

                doc_id = add_document_sync(
                    content=workflow_content,
                    metadata={
                        'source': str(file_path),
                        'type': 'product_workflow',
                        'workflow_id': workflow_id,
                        'workflow_name': workflow_name,
                        'product': product_name
                    }
                )
                if doc_id:
                    document_ids.append(doc_id)
                    chunk_count += 1

            record.document_ids = document_ids
            record.num_chunks = chunk_count

            logger.info(f"Successfully ingested product catalog: {chunk_count} documents created")
            return True

        except Exception as e:
            logger.error(f"Failed to ingest product catalog {file_path}: {e}", exc_info=True)
            return False


_monitor_instance: Optional[DataFileMonitor] = None


def get_monitor() -> DataFileMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = DataFileMonitor()
    return _monitor_instance


def start_monitoring():
    """Start the data file monitoring system"""
    monitor = get_monitor()
    monitor.start()
    return monitor


def stop_monitoring():
    """Stop the data file monitoring system"""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop()
        _monitor_instance = None
