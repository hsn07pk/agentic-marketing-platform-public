import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from sqlalchemy import select, text, delete, MetaData, Table
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer

from ..database.connection import get_async_session
from ...config.settings import settings

logger = logging.getLogger(__name__)

class PgVectorStore:
    """
    Vector store using PostgreSQL with pgvector extension
    Used for RAG retrieval and semantic search
    """

    _shared_embedding_model = None

    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.embedding_dim = 384

    @property
    def embedding_model(self):
        """Lazy-load embedding model (shared across instances)"""
        if PgVectorStore._shared_embedding_model is None:
            logger.info("Loading SentenceTransformer model (first use)...")
            PgVectorStore._shared_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return PgVectorStore._shared_embedding_model
    
    async def initialize(self):
        """Initialize vector store table"""
        async with get_async_session() as session:
            await session.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({self.embedding_dim}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            await session.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {self.collection_name}_embedding_idx 
                ON {self.collection_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            
            await session.commit()
            logger.info(f"Initialized vector store: {self.collection_name}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    
    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add document to vector store
        
        Args:
            content: Document text
            metadata: Optional metadata
        
        Returns:
            Document ID
        """
        try:
            embedding = self.embed_text(content)
            embedding_list = embedding.tolist()

            import json
            async with get_async_session() as session:
                embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
                metadata_str = json.dumps(metadata or {})

                query = f"""
                    INSERT INTO {self.collection_name}
                    (content, embedding, metadata)
                    VALUES (:content, :embedding, :metadata)
                    RETURNING id
                """

                result = await session.execute(
                    text(query),
                    {"content": content, "embedding": embedding_str, "metadata": metadata_str}
                )

                doc_id = result.scalar_one()
                await session.commit()
                
                logger.info(f"Added document {doc_id} to {self.collection_name}")
                return doc_id
        
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    async def add_documents(
        self,
        documents: List[Any]
    ) -> List[int]:
        """
        Add multiple documents

        Args:
            documents: List of dicts with 'content' and optional 'metadata',
                      or LangChain Document objects

        Returns:
            List of document IDs
        """
        doc_ids = []

        for doc in documents:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            elif isinstance(doc, dict):
                content = doc['content']
                metadata = doc.get('metadata', {})
            else:
                logger.warning(f"Unknown document type: {type(doc)}, skipping")
                continue

            doc_id = await self.add_document(
                content=content,
                metadata=metadata
            )
            doc_ids.append(doc_id)

        return doc_ids
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            top_k: Number of results
            filter_metadata: Optional metadata filter
        
        Returns:
            List of similar documents with scores
        """
        try:
            query_embedding = self.embed_text(query)
            query_embedding_list = query_embedding.tolist()

            import json
            async with get_async_session() as session:
                # Set ivfflat.probes to search more index partitions for better recall
                # With lists=100, probes=100 searches all partitions (brute force for small datasets)
                await session.execute(text("SET ivfflat.probes = 100"))

                if filter_metadata:
                    metadata_filter = " AND " + " AND ".join([
                        f"metadata->>'{k}' = '{v}'"
                        for k, v in filter_metadata.items()
                    ])
                else:
                    metadata_filter = ""

                query_embedding_str = '[' + ','.join(map(str, query_embedding_list)) + ']'

                # Embed vector string directly in SQL (safe since we generate it, not user input)
                # Using parameter binding with ::vector cast doesn't work in PostgreSQL
                sql_query = f"""
                    SELECT
                        id,
                        content,
                        metadata,
                        created_at,
                        1 - (embedding <=> '{query_embedding_str}'::vector) as similarity
                    FROM {self.collection_name}
                    WHERE 1=1 {metadata_filter}
                    ORDER BY embedding <=> '{query_embedding_str}'::vector
                    LIMIT :top_k
                """

                result = await session.execute(
                    text(sql_query),
                    {"top_k": top_k}
                )
                
                documents = []
                for row in result:
                    documents.append({
                        "id": row.id,
                        "content": row.content,
                        "metadata": row.metadata,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "similarity": float(row.similarity)
                    })
                
                logger.info(f"Found {len(documents)} similar documents")
                return documents
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def delete_document(self, doc_id: int) -> bool:
        """
        Delete document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            Success boolean (True if document was found and deleted)
        """
        try:
            async with get_async_session() as session:
                query = f"""
                    DELETE FROM {self.collection_name}
                    WHERE id = :doc_id
                """
                result = await session.execute(
                    text(query),
                    {"doc_id": doc_id}
                )
                
                await session.commit()
                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"Deleted document {doc_id}")
                else:
                    logger.warning(f"Document {doc_id} not found for deletion")
                return deleted
        
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    async def delete_by_metadata(self, filter_metadata: Dict[str, Any]) -> int:
        """
        Delete documents matching metadata filter
        
        Args:
            filter_metadata: Metadata key-value pairs to match
        
        Returns:
            Number of documents deleted
        """
        try:
            async with get_async_session() as session:
                conditions = " AND ".join([
                    f"metadata->>'{k}' = '{v}'"
                    for k, v in filter_metadata.items()
                ])
                
                count_query = f"""
                    SELECT COUNT(*) as total
                    FROM {self.collection_name}
                    WHERE {conditions}
                """
                count_result = await session.execute(text(count_query))
                count_row = count_result.first()
                count = count_row.total if count_row else 0
                
                delete_query = f"""
                    DELETE FROM {self.collection_name}
                    WHERE {conditions}
                """
                await session.execute(text(delete_query))
                await session.commit()
                
                logger.info(f"Deleted {count} documents matching filter {filter_metadata}")
                return count
        
        except Exception as e:
            logger.error(f"Failed to delete by metadata: {e}")
            return 0
    
    async def clear_collection(self) -> bool:
        """
        Clear all documents from collection
        
        Returns:
            Success boolean
        """
        try:
            async with get_async_session() as session:
                await session.execute(
                    text(f"TRUNCATE TABLE {self.collection_name}")
                )
                
                await session.commit()
                logger.info(f"Cleared collection {self.collection_name}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    async def count_documents(self, category: Optional[str] = None) -> int:
        """
        Get total count of documents in collection

        Args:
            category: Optional category filter
            
        Returns:
            Total document count
        """
        try:
            async with get_async_session() as session:
                where_clause = ""
                params = {}
                
                if category and category.lower() != 'all':
                    where_clause = "WHERE metadata->>'category' ILIKE :category"
                    params["category"] = category
                
                result = await session.execute(
                    text(f"""
                        SELECT COUNT(*) as total
                        FROM {self.collection_name}
                        {where_clause}
                    """),
                    params
                )

                row = result.first()
                return row.total if row else 0

        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics

        Returns:
            Collection stats
        """
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    text(f"""
                        SELECT COUNT(*) as total_docs
                        FROM {self.collection_name}
                    """)
                )

                row = result.first()

                return {
                    "collection_name": self.collection_name,
                    "total_documents": row.total_docs if row else 0,
                    "embedding_dimension": self.embedding_dim
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "embedding_dimension": self.embedding_dim
            }

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        LangChain-compatible similarity search
        Returns list of Document-like objects
        """
        try:
            results = await self.search(query, top_k=k, filter_metadata=filter)

            from langchain.schema import Document
            documents = []
            for result in results:
                doc = Document(
                    page_content=result["content"],
                    metadata=result.get("metadata", {})
                )
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error(f"similarity_search failed: {e}")
            return []

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Any, float]]:
        """
        LangChain-compatible similarity search with scores
        Returns list of (Document, score) tuples
        """
        try:
            results = await self.search(query, top_k=k, filter_metadata=filter)

            from langchain.schema import Document
            documents_with_scores = []
            for result in results:
                doc = Document(
                    page_content=result["content"],
                    metadata=result.get("metadata", {})
                )
                score = result.get("similarity", 0.0)
                documents_with_scores.append((doc, score))

            return documents_with_scores
        except Exception as e:
            logger.error(f"similarity_search_with_score failed: {e}")
            return []

    async def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a single document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document dict with id, content, metadata, created_at or None
        """
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    text(f"""
                        SELECT id, content, metadata, created_at
                        FROM {self.collection_name}
                        WHERE id = :doc_id
                    """),
                    {"doc_id": doc_id}
                )
                
                row = result.first()
                if row:
                    return {
                        "id": row.id,
                        "content": row.content,
                        "metadata": row.metadata or {},
                        "created_at": row.created_at.isoformat() if row.created_at else None
                    }
                return None
        
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    async def get_document_with_embedding(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a single document by ID including the raw embedding vector
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document dict with id, content, metadata, created_at, embedding
        """
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    text(f"""
                        SELECT id, content, metadata, created_at, embedding::text as embedding_str
                        FROM {self.collection_name}
                        WHERE id = :doc_id
                    """),
                    {"doc_id": doc_id}
                )
                
                row = result.first()
                if row:
                    embedding = None
                    if row.embedding_str:
                        embedding_str = row.embedding_str.strip('[]')
                        if embedding_str:
                            embedding = [float(x) for x in embedding_str.split(',')]
                    
                    return {
                        "id": row.id,
                        "content": row.content,
                        "metadata": row.metadata or {},
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "embedding": embedding,
                        "embedding_dim": len(embedding) if embedding else 0
                    }
                return None
        
        except Exception as e:
            logger.error(f"Failed to get document with embedding {doc_id}: {e}")
            return None

    async def update_document(
        self,
        doc_id: int,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update document content and/or metadata. If content is changed, re-embeds.
        
        Args:
            doc_id: Document ID
            content: New content (triggers re-embedding)
            metadata: New metadata dict
        
        Returns:
            Success boolean
        """
        try:
            import json
            async with get_async_session() as session:
                updates = []
                params = {"doc_id": doc_id}
                
                if content is not None:
                    embedding = self.embed_text(content)
                    embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
                    updates.append("content = :content")
                    updates.append("embedding = :embedding")
                    params["content"] = content
                    params["embedding"] = embedding_str
                
                if metadata is not None:
                    updates.append("metadata = :metadata")
                    params["metadata"] = json.dumps(metadata)
                
                if not updates:
                    return True
                
                query = f"""
                    UPDATE {self.collection_name}
                    SET {', '.join(updates)}
                    WHERE id = :doc_id
                """
                
                await session.execute(text(query), params)
                await session.commit()
                
                logger.info(f"Updated document {doc_id}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False

    async def update_metadata(self, doc_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Update only document metadata without re-embedding
        
        Args:
            doc_id: Document ID
            metadata: New metadata dict
        
        Returns:
            Success boolean
        """
        try:
            import json
            async with get_async_session() as session:
                await session.execute(
                    text(f"""
                        UPDATE {self.collection_name}
                        SET metadata = :metadata
                        WHERE id = :doc_id
                    """),
                    {"doc_id": doc_id, "metadata": json.dumps(metadata)}
                )
                await session.commit()
                
                logger.info(f"Updated metadata for document {doc_id}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to update metadata for {doc_id}: {e}")
            return False

    async def list_documents(
        self,
        limit: int = 20,
        offset: int = 0,
        category: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        List documents with pagination and filtering
        
        Args:
            limit: Max documents to return
            offset: Number of documents to skip
            category: Optional category filter
            sort_by: Field to sort by (id, created_at)
            sort_order: asc or desc
        
        Returns:
            List of document dicts
        """
        try:
            async with get_async_session() as session:
                where_clause = ""
                params = {"limit": limit, "offset": offset}
                
                if category and category.lower() != 'all':
                    where_clause = "WHERE metadata->>'category' ILIKE :category"
                    params["category"] = category
                
                # Validate sort_by to prevent SQL injection
                valid_sort_fields = {
                    "id": "id",
                    "created_at": "created_at",
                    "title": "metadata->>'title'",
                    "category": "metadata->>'category'",
                    "word_count": "LENGTH(content)"
                }
                sort_column = valid_sort_fields.get(sort_by, "created_at")
                
                sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
                
                result = await session.execute(
                    text(f"""
                        SELECT id, content, metadata, created_at
                        FROM {self.collection_name}
                        {where_clause}
                        ORDER BY {sort_column} {sort_direction}
                        LIMIT :limit OFFSET :offset
                    """),
                    params
                )
                
                documents = []
                for row in result:
                    meta = row.metadata or {}
                    documents.append({
                        "id": row.id,
                        "title": meta.get("title", f"Document {row.id}"),
                        "content": row.content[:200] + "..." if len(row.content) > 200 else row.content,
                        "full_content": row.content,
                        "metadata": meta,
                        "category": meta.get("category", "general"),
                        "tags": meta.get("tags", []),
                        "word_count": len(row.content.split()),
                        "created_at": row.created_at.isoformat() if row.created_at else None
                    })
                
                return documents
        
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    async def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all vector collections (tables with embedding column) in the database
        
        Returns:
            List of collection info dicts
        """
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    text("""
                        SELECT table_name
                        FROM information_schema.columns
                        WHERE column_name = 'embedding'
                        AND data_type = 'USER-DEFINED'
                        AND udt_name = 'vector'
                        AND table_schema = 'public'
                    """)
                )
                
                collections = []
                for row in result:
                    table_name = row.table_name
                    count_result = await session.execute(
                        text(f"SELECT COUNT(*) as total FROM {table_name}")
                    )
                    count_row = count_result.first()
                    
                    collections.append({
                        "name": table_name,
                        "total_documents": count_row.total if count_row else 0
                    })
                
                return collections
        
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    async def find_similar(self, doc_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document
        
        Args:
            doc_id: Source document ID
            top_k: Number of similar documents to return
        
        Returns:
            List of similar documents with similarity scores
        """
        try:
            async with get_async_session() as session:
                source_result = await session.execute(
                    text(f"""
                        SELECT embedding::text as embedding_str
                        FROM {self.collection_name}
                        WHERE id = :doc_id
                    """),
                    {"doc_id": doc_id}
                )
                
                source_row = source_result.first()
                if not source_row or not source_row.embedding_str:
                    return []
                
                embedding_str = source_row.embedding_str
                
                await session.execute(text("SET ivfflat.probes = 100"))
                
                result = await session.execute(
                    text(f"""
                        SELECT id, content, metadata,
                               1 - (embedding <=> '{embedding_str}'::vector) as similarity
                        FROM {self.collection_name}
                        WHERE id != :doc_id
                        ORDER BY embedding <=> '{embedding_str}'::vector
                        LIMIT :fetch_limit
                    """),
                    {"doc_id": doc_id, "fetch_limit": top_k * 3}  # Fetch extra for dedup
                )
                
                documents = []
                seen_titles = set()
                seen_content_hashes = set()
                
                for row in result:
                    meta = row.metadata or {}
                    title = meta.get("title", f"Document {row.id}")
                    
                    if float(row.similarity) > 0.99:

                        continue
                    
                    title_lower = title.lower().strip()
                    if title_lower in seen_titles:
                        continue
                    seen_titles.add(title_lower)
                    
                    content_hash = hash(row.content[:200] if row.content else "")
                    if content_hash in seen_content_hashes:
                        continue
                    seen_content_hashes.add(content_hash)
                    
                    documents.append({
                        "id": row.id,
                        "title": title,
                        "content": row.content[:200] + "..." if len(row.content) > 200 else row.content,
                        "metadata": meta,
                        "similarity": float(row.similarity)
                    })
                    
                    if len(documents) >= top_k:
                        break
                
                return documents
        
        except Exception as e:
            logger.error(f"Failed to find similar documents for {doc_id}: {e}")
            return []

    async def get_embeddings_for_visualization(
        self,
        limit: int = 100,
        reduce_dimensions: bool = True
    ) -> Dict[str, Any]:
        """
        Get document embeddings for 2D/3D visualization
        
        Args:
            limit: Max number of documents to include
            reduce_dimensions: Whether to apply t-SNE reduction
        
        Returns:
            Dict with points (x, y coordinates), labels, and metadata
        """
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    text(f"""
                        SELECT id, metadata, embedding::text as embedding_str
                        FROM {self.collection_name}
                        ORDER BY RANDOM()
                        LIMIT :limit
                    """),
                    {"limit": limit}
                )
                
                embeddings = []
                doc_ids = []
                titles = []
                categories = []
                
                for row in result:
                    if row.embedding_str:
                        embedding_str = row.embedding_str.strip('[]')
                        if embedding_str:
                            embedding = [float(x) for x in embedding_str.split(',')]
                            embeddings.append(embedding)
                            doc_ids.append(row.id)
                            meta = row.metadata or {}
                            title = (
                                meta.get("title") or 
                                meta.get("product") or 
                                meta.get("name") or 
                                meta.get("source", "").split("/")[-1] or
                                f"Doc {row.id}"
                            )
                            titles.append(title[:50] + "..." if len(title) > 50 else title)
                            categories.append(meta.get("category", "general"))
                
                if not embeddings:
                    return {"points": [], "doc_ids": [], "titles": [], "categories": []}
                
                if reduce_dimensions and len(embeddings) >= 5:
                    try:
                        from sklearn.manifold import TSNE
                        
                        perplexity = min(30, len(embeddings) - 1)
                        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                        reduced = tsne.fit_transform(np.array(embeddings))
                        
                        points = [{"x": float(p[0]), "y": float(p[1])} for p in reduced]
                    except Exception as tsne_error:
                        logger.warning(f"t-SNE failed, returning raw first 2 dims: {tsne_error}")
                        points = [{"x": float(e[0]), "y": float(e[1])} for e in embeddings]
                else:
                    points = [{"x": float(e[0]), "y": float(e[1])} for e in embeddings]
                
                return {
                    "points": points,
                    "doc_ids": doc_ids,
                    "titles": titles,
                    "categories": categories,
                    "total": len(points)
                }
        
        except Exception as e:
            logger.error(f"Failed to get embeddings for visualization: {e}")
            return {"points": [], "doc_ids": [], "titles": [], "categories": [], "error": str(e)}