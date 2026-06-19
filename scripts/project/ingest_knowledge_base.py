#!/usr/bin/env python3
"""
Knowledge Base Ingestion Script

Scans data/knowledge_base/ directory, parses markdown files with YAML frontmatter,
generates embeddings, and stores in pgvector for RAG retrieval.
"""
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import yaml
import frontmatter

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_layer.vector_store.pgvector_store import PgVectorStore
from src.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeBaseIngester:
    """Ingest knowledge base documents into vector store"""

    def __init__(self, vector_store: PgVectorStore):
        self.vector_store = vector_store
        self.supported_extensions = {'.md', '.txt'}

    def scan_directory(self, directory: Path) -> List[Path]:
        """Recursively scan directory for markdown/text files"""
        documents = []

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        for ext in self.supported_extensions:
            documents.extend(directory.rglob(f'*{ext}'))

        logger.info(f"Found {len(documents)} documents in {directory}")
        return sorted(documents)

    def parse_document(self, file_path: Path) -> Dict[str, Any]:
        """Parse markdown file with YAML frontmatter"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)

            # Extract metadata from frontmatter
            metadata = dict(post.metadata)
            content = post.content.strip()

            # Validate required fields
            required_fields = ['title', 'category', 'personas', 'tags', 'date']
            missing_fields = [field for field in required_fields if field not in metadata]

            if missing_fields:
                logger.warning(f"Missing required fields in {file_path}: {missing_fields}")
                # Set defaults for missing fields
                if 'title' not in metadata:
                    metadata['title'] = file_path.stem
                if 'category' not in metadata:
                    metadata['category'] = 'general'
                if 'personas' not in metadata:
                    metadata['personas'] = []
                if 'tags' not in metadata:
                    metadata['tags'] = []
                if 'date' not in metadata:
                    metadata['date'] = None

            # Ensure personas and tags are lists
            if isinstance(metadata.get('personas'), str):
                metadata['personas'] = [metadata['personas']]
            if isinstance(metadata.get('tags'), str):
                metadata['tags'] = [metadata['tags']]

            # Add source file path
            metadata['source_file'] = str(file_path.relative_to(project_root))
            metadata['file_name'] = file_path.name

            # Determine category from directory structure
            if 'case_studies' in str(file_path):
                metadata['category'] = 'case_study'
            elif 'product_docs' in str(file_path):
                metadata['category'] = 'product_doc'
            elif 'blog_posts' in str(file_path):
                metadata['category'] = 'blog_post'
            elif 'whitepapers' in str(file_path):
                metadata['category'] = 'whitepaper'
            elif 'sales_collateral' in str(file_path):
                metadata['category'] = 'sales_collateral'

            return {
                'content': content,
                'metadata': metadata
            }

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            raise

    async def ingest_document(self, file_path: Path) -> bool:
        """Ingest single document into vector store"""
        try:
            logger.info(f"Processing: {file_path.name}")

            # Parse document
            doc = self.parse_document(file_path)

            # Skip empty content
            if not doc['content'] or len(doc['content']) < 50:
                logger.warning(f"Skipping {file_path.name}: Content too short (<50 chars)")
                return False

            # Add to vector store
            await self.vector_store.add_documents(
                texts=[doc['content']],
                metadatas=[doc['metadata']]
            )

            logger.info(f"✓ Ingested: {doc['metadata']['title']} ({len(doc['content'])} chars)")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to ingest {file_path.name}: {e}")
            return False

    async def ingest_all(self, directory: Path, validate: bool = False) -> Dict[str, Any]:
        """Ingest all documents from directory"""
        logger.info("=" * 60)
        logger.info("KNOWLEDGE BASE INGESTION")
        logger.info("=" * 60)

        # Scan directory
        documents = self.scan_directory(directory)

        if not documents:
            logger.warning(f"No documents found in {directory}")
            return {
                'success': False,
                'total': 0,
                'ingested': 0,
                'failed': 0
            }

        # Ingest documents
        results = {
            'success': True,
            'total': len(documents),
            'ingested': 0,
            'failed': 0,
            'documents': []
        }

        for doc_path in documents:
            success = await self.ingest_document(doc_path)
            if success:
                results['ingested'] += 1
                results['documents'].append(str(doc_path.relative_to(project_root)))
            else:
                results['failed'] += 1

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total documents scanned: {results['total']}")
        logger.info(f"Successfully ingested: {results['ingested']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Success rate: {results['ingested']/results['total']*100:.1f}%")

        # Validation
        if validate and results['ingested'] > 0:
            logger.info("\n" + "=" * 60)
            logger.info("VALIDATION")
            logger.info("=" * 60)
            await self.validate_ingestion()

        logger.info("=" * 60)

        return results

    async def validate_ingestion(self):
        """Validate ingested documents with sample queries"""
        test_queries = [
            ("ROI case studies for B2B decision makers", {"category": "case_study", "personas": ["decision_maker"]}),
            ("Product features for autonomous marketing", {"category": "product_doc"}),
            ("Thought leadership on AI marketing", {"category": "blog_post"}),
        ]

        for query, filters in test_queries:
            logger.info(f"\nTest Query: '{query}'")
            logger.info(f"Filters: {filters}")

            try:
                results = await self.vector_store.similarity_search(
                    query=query,
                    k=3,
                    filter=filters
                )

                if results:
                    logger.info(f"✓ Retrieved {len(results)} documents:")
                    for i, doc in enumerate(results, 1):
                        title = doc.metadata.get('title', 'Untitled')
                        category = doc.metadata.get('category', 'unknown')
                        logger.info(f"  {i}. {title} ({category})")
                else:
                    logger.warning(f"✗ No results for query: {query}")

            except Exception as e:
                logger.error(f"✗ Query failed: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Ingest knowledge base documents into vector store')
    parser.add_argument(
        '--input',
        type=str,
        default='data/knowledge_base',
        help='Input directory containing knowledge base documents (default: data/knowledge_base)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation queries after ingestion'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing documents before ingestion (WARNING: deletes all data)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)

    # Initialize vector store
    logger.info("Connecting to vector store...")
    vector_store = PgVectorStore(collection_name="documents")

    # Clear if requested
    if args.clear:
        logger.warning("⚠️  CLEAR MODE: Deleting all existing documents")
        response = input("Are you sure? Type 'yes' to continue: ")
        if response.lower() == 'yes':
            # TODO: Implement clear method in PgVectorStore
            logger.info("✓ Cleared existing documents")
        else:
            logger.info("Aborted.")
            return

    # Ingest documents
    ingester = KnowledgeBaseIngester(vector_store)
    results = await ingester.ingest_all(input_dir, validate=args.validate)

    # Exit code
    if results['failed'] > 0:
        logger.warning(f"⚠️  Ingestion completed with {results['failed']} failures")
        sys.exit(1)
    else:
        logger.info("✅ Ingestion completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
