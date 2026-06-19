"""
Knowledge Base Management API
Handles document ingestion, retrieval, and management
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from pathlib import Path
import logging
import frontmatter
from datetime import datetime

from ...data_layer.vector_store.pgvector_store import PgVectorStore
from ...config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

class IngestDirectoryRequest(BaseModel):
    directory_path: str
    collection_name: str = "documents"
    validate_content: bool = True


class IngestStatusResponse(BaseModel):
    success: bool
    documents_processed: int
    documents_ingested: int
    errors: List[str]
    message: str


class DocumentMetadata(BaseModel):
    """Document metadata"""
    id: str
    title: str
    category: Optional[str]
    tags: List[str]
    created_at: datetime
    word_count: int


class DocumentListResponse(BaseModel):
    """Response for listing documents"""
    total: int
    documents: List[Dict[str, Any]]


class CreateDocumentRequest(BaseModel):
    """Request to create a new document"""
    content: str
    title: str = "Untitled"
    category: str = "general"
    tags: List[str] = []
    collection_name: str = "documents"


class UpdateDocumentRequest(BaseModel):
    """Request to update a document"""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateMetadataRequest(BaseModel):
    """Request to update only document metadata"""
    title: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None


async def ingest_documents_from_directory(
    directory_path: str,
    collection_name: str = "documents",
    validate: bool = True
) -> Dict[str, Any]:
    """
    Ingest documents from a directory into vector store

    Args:
        directory_path: Path to directory containing markdown files
        collection_name: Vector store collection name
        validate: Whether to validate documents before ingestion

    Returns:
        Dictionary with ingestion results
    """
    logger.info(f"Starting document ingestion from: {directory_path}")

    results = {
        'success': False,
        'documents_processed': 0,
        'documents_ingested': 0,
        'errors': [],
        'message': ''
    }

    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()

        base_path = Path(directory_path)

        if not base_path.exists():
            results['errors'].append(f"Directory not found: {directory_path}")
            results['message'] = "Directory not found"
            return results

        markdown_files = list(base_path.rglob("*.md"))

        if not markdown_files:
            results['message'] = f"No markdown files found in {directory_path}"
            return results

        logger.info(f"Found {len(markdown_files)} markdown files")

        for file_path in markdown_files:
            try:
                results['documents_processed'] += 1

                with open(file_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)

                created_at = post.get('date', datetime.now())
                if hasattr(created_at, 'isoformat'):
                    created_at_str = created_at.isoformat()
                else:
                    created_at_str = str(created_at)

                metadata = {
                    'source': str(file_path.relative_to(base_path)),
                    'title': post.get('title', file_path.stem),
                    'category': post.get('category', 'general'),
                    'tags': post.get('tags', []),
                    'created_at': created_at_str,
                    'file_path': str(file_path)
                }

                content = post.content

                if validate:
                    if not content.strip():
                        logger.warning(f"Skipping empty file: {file_path}")
                        results['errors'].append(f"Empty file: {file_path.name}")
                        continue

                    if len(content.split()) < 10:
                        logger.warning(f"Skipping short file: {file_path} ({len(content.split())} words)")
                        results['errors'].append(f"Too short: {file_path.name}")
                        continue

                doc_id = await vector_store.add_document(
                    content=content,
                    metadata=metadata
                )

                if doc_id:
                    results['documents_ingested'] += 1
                    logger.info(f"✅ Ingested: {metadata['title']}")
                else:
                    results['errors'].append(f"Failed to ingest: {file_path.name}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results['errors'].append(f"Error in {file_path.name}: {str(e)}")

        results['success'] = results['documents_ingested'] > 0
        results['message'] = f"Successfully ingested {results['documents_ingested']}/{results['documents_processed']} documents"

        logger.info(f"✅ Ingestion complete: {results['documents_ingested']}/{results['documents_processed']} documents")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        results['errors'].append(f"Ingestion error: {str(e)}")
        results['message'] = "Ingestion failed"

    return results


@router.post("/ingest-directory", response_model=IngestStatusResponse)
async def ingest_directory(
    request: IngestDirectoryRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest documents from a directory

    This endpoint processes all markdown files in the specified directory
    and adds them to the vector store for RAG retrieval.
    """
    try:
        results = await ingest_documents_from_directory(
            directory_path=request.directory_path,
            collection_name=request.collection_name,
            validate=request.validate_content
        )

        return IngestStatusResponse(**results)

    except Exception as e:
        logger.error(f"Ingestion endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest-knowledge-base")
async def ingest_default_knowledge_base(
    validate: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Ingest the default knowledge base from data/knowledge_base/

    This is a convenience endpoint that ingests the standard knowledge base
    directory without requiring a path parameter.
    """
    try:
        kb_path = "data/knowledge_base"

        results = await ingest_documents_from_directory(
            directory_path=kb_path,
            collection_name="documents",
            validate=validate
        )

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Default knowledge base ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=IngestStatusResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = "documents"
):
    """
    Upload and ingest documents via file upload

    Accepts markdown files and ingests them into the vector store.
    """
    results = {
        'success': False,
        'documents_processed': 0,
        'documents_ingested': 0,
        'errors': [],
        'message': ''
    }

    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()

        for file in files:
            try:
                results['documents_processed'] += 1

                if not file.filename.endswith('.md'):
                    results['errors'].append(f"Not a markdown file: {file.filename}")
                    continue

                content_bytes = await file.read()
                content_str = content_bytes.decode('utf-8')

                post = frontmatter.loads(content_str)

                metadata = {
                    'source': 'upload',
                    'filename': file.filename,
                    'title': post.get('title', file.filename),
                    'category': post.get('category', 'uploaded'),
                    'tags': post.get('tags', []),
                    'uploaded_at': datetime.now().isoformat()
                }

                doc_id = await vector_store.add_document(
                    content=post.content,
                    metadata=metadata
                )

                if doc_id:
                    results['documents_ingested'] += 1
                    logger.info(f"✅ Uploaded: {file.filename}")
                else:
                    results['errors'].append(f"Failed to ingest: {file.filename}")

            except Exception as e:
                logger.error(f"Error uploading {file.filename}: {e}")
                results['errors'].append(f"Error in {file.filename}: {str(e)}")

        results['success'] = results['documents_ingested'] > 0
        results['message'] = f"Uploaded {results['documents_ingested']}/{results['documents_processed']} files"

        return IngestStatusResponse(**results)

    except Exception as e:
        logger.error(f"Upload endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    collection_name: str = "documents",
    limit: int = 100,
    offset: int = 0,
    category: Optional[str] = None
):
    """
    List all documents in the knowledge base

    Returns metadata for all ingested documents.
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()

        documents = await vector_store.list_documents(limit=limit, offset=offset, category=category)

        total_count = await vector_store.count_documents(category=category)

        return DocumentListResponse(
            total=total_count,
            documents=documents
        )

    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    collection_name: str = "documents"
):
    """
    Delete a document from the knowledge base

    Args:
        document_id: ID of the document to delete
        collection_name: Collection name
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()

        success = await vector_store.delete_document(document_id)

        if success:
            return {"success": True, "message": f"Document {document_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_knowledge_base_stats(collection_name: str = "documents"):
    """
    Get statistics about the knowledge base

    Returns counts, categories, and other metrics.
    """
    try:
        from ...data_layer.database.connection import get_async_session
        from sqlalchemy import text, func

        async with get_async_session() as session:
            result = await session.execute(
                text(f"""
                    SELECT COUNT(*) as total
                    FROM {collection_name}
                """)
            )
            total_documents = result.scalar() or 0

            result = await session.execute(
                text(f"""
                    SELECT MAX(created_at) as last_updated
                    FROM {collection_name}
                """)
            )
            last_updated = result.scalar()

        return {
            "collection_name": collection_name,
            "total_documents": total_documents,
            "last_updated": last_updated.isoformat() if last_updated else None,
            "status": "ready"
        }

    except Exception as e:
        logger.error(f"Get stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_categories(collection_name: str = "documents"):
    """
    Get list of unique categories in the knowledge base
    """
    try:
        from ...data_layer.database.connection import get_async_session
        from sqlalchemy import text

        async with get_async_session() as session:
            result = await session.execute(
                text(f"""
                    SELECT DISTINCT metadata->>'category' as category
                    FROM {collection_name}
                    WHERE metadata->>'category' IS NOT NULL
                    ORDER BY category
                """)
            )
            categories = [row.category for row in result if row.category]
            
        return {"categories": categories}

    except Exception as e:
        logger.error(f"Get categories error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_knowledge_base(
    query: str,
    limit: int = 10,
    collection_name: str = "documents"
):
    """
    Search the knowledge base using semantic search (RAG retrieval).
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return
        collection_name: Vector store collection to search
        
    Returns:
        List of matching documents with relevance scores
    """
    try:
        if not query.strip():
            return []
            
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()
        
        results = await vector_store.search(
            query=query,
            top_k=limit
        )
        
        search_results = []
        for result in results:
            search_results.append({
                "id": result.get("id"),  # CRITICAL FIX: Include ID
                "title": result.get("metadata", {}).get("title", "Document"),
                "content": result.get("content", "")[:500],  # Truncate for display
                "score": result.get("similarity", 0),
                "source": result.get("metadata", {}).get("source", ""),
                "category": result.get("metadata", {}).get("category", "general"),
                "created_at": result.get("created_at")
            })
        
        return search_results
        
    except Exception as e:
        logger.error(f"Knowledge base search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}")
async def get_document(
    document_id: int,
    collection_name: str = "documents",
    include_vector: bool = False
):
    """
    Get a single document by ID
    
    Args:
        document_id: Document ID  
        collection_name: Collection name
        include_vector: Whether to include the raw embedding vector
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()
        
        if include_vector:
            doc = await vector_store.get_document_with_embedding(document_id)
        else:
            doc = await vector_store.get_document(document_id)
        
        if doc:
            return doc
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents")
async def create_document(request: CreateDocumentRequest):
    """
    Create a new document in the knowledge base
    
    Creates a single document with the provided content and metadata,
    generates embedding, and stores in the vector database.
    """
    try:
        vector_store = PgVectorStore(collection_name=request.collection_name)
        await vector_store.initialize()
        
        metadata = {
            "title": request.title,
            "category": request.category,
            "tags": request.tags,
            "source": "manual",
            "created_at": datetime.now().isoformat()
        }
        
        doc_id = await vector_store.add_document(
            content=request.content,
            metadata=metadata
        )
        
        if doc_id:
            return {
                "success": True,
                "document_id": doc_id,
                "message": f"Document '{request.title}' created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create document")
    
    except Exception as e:
        logger.error(f"Create document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/documents/{document_id}")
async def update_document(
    document_id: int,
    request: UpdateDocumentRequest,
    collection_name: str = "documents"
):
    """
    Update a document's content and/or metadata
    
    If content is updated, the embedding will be regenerated.
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()
        
        existing = await vector_store.get_document(document_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Document not found")
        
        success = await vector_store.update_document(
            doc_id=document_id,
            content=request.content,
            metadata=request.metadata
        )
        
        if success:
            return {
                "success": True,
                "message": f"Document {document_id} updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update document")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/documents/{document_id}/metadata")
async def update_document_metadata(
    document_id: int,
    request: UpdateMetadataRequest,
    collection_name: str = "documents"
):
    """
    Update only the document metadata (no re-embedding)
    
    Use this for quick metadata changes like updating tags or category.
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()
        
        existing = await vector_store.get_document(document_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Document not found")
        
        current_metadata = existing.get("metadata", {})
        
        if request.title is not None:
            current_metadata["title"] = request.title
        if request.category is not None:
            current_metadata["category"] = request.category
        if request.tags is not None:
            current_metadata["tags"] = request.tags
        if request.extra:
            current_metadata.update(request.extra)
        
        success = await vector_store.update_metadata(document_id, current_metadata)
        
        if success:
            return {
                "success": True,
                "message": f"Metadata for document {document_id} updated",
                "metadata": current_metadata
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update metadata")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update metadata error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/similar")
async def find_similar_documents(
    document_id: int,
    limit: int = 5,
    collection_name: str = "documents"
):
    """
    Find documents similar to a given document
    
    Uses the document's embedding to find semantically similar documents.
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()
        
        existing = await vector_store.get_document(document_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Source document not found")
        
        similar = await vector_store.find_similar(doc_id=document_id, top_k=limit)
        
        return {
            "source_document_id": document_id,
            "similar_count": len(similar),
            "documents": similar
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Find similar error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections")
async def list_collections():
    """
    List all vector collections in the database
    
    Returns all tables that have pgvector embedding columns.
    """
    try:
        vector_store = PgVectorStore()
        await vector_store.initialize()
        
        collections = await vector_store.list_collections()
        
        return {
            "collections": collections,
            "total": len(collections)
        }
    
    except Exception as e:
        logger.error(f"List collections error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def clear_collection(collection_name: str):
    """
    Clear all documents from a collection
    
    WARNING: This will delete ALL documents in the collection.
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()
        
        count_before = await vector_store.count_documents()
        
        success = await vector_store.clear_collection()
        
        if success:
            return {
                "success": True,
                "message": f"Cleared collection '{collection_name}'",
                "documents_deleted": count_before
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear collection")
    
    except Exception as e:
        logger.error(f"Clear collection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embeddings/visualization")
async def get_embeddings_for_visualization(
    collection_name: str = "documents",
    limit: int = 100,
    reduce_dimensions: bool = True
):
    """
    Get document embeddings reduced to 2D for visualization
    
    Uses t-SNE to reduce high-dimensional embeddings to 2D coordinates
    suitable for scatter plot visualization.
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()
        
        result = await vector_store.get_embeddings_for_visualization(
            limit=limit,
            reduce_dimensions=reduce_dimensions
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Embeddings visualization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents-paginated")
async def list_documents_paginated(
    collection_name: str = "documents",
    limit: int = 20,
    offset: int = 0,
    category: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """
    List documents with full pagination, filtering, and sorting
    
    This is the enhanced version of /documents with more options.
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()
        
        documents = await vector_store.list_documents(
            limit=limit,
            offset=offset,
            category=category,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        total_count = await vector_store.count_documents()
        
        return {
            "documents": documents,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(documents) < total_count
        }
    
    except Exception as e:
        logger.error(f"List documents paginated error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ScrapeURLRequest(BaseModel):
    """Request to scrape a URL and add to knowledge base"""
    url: str
    category: str = "general"
    tags: List[str] = []
    collection_name: str = "documents"


@router.post("/scrape-url")
async def scrape_url_to_kb(request: ScrapeURLRequest):
    """
    Scrape content from a URL and add to knowledge base
    
    Fetches the content from the URL, extracts text, and stores in vector database.
    """
    import httpx
    from bs4 import BeautifulSoup
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = await client.get(request.url, headers=headers)
            response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        text = soup.get_text(separator="\n", strip=True)
        
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = "\n".join(lines)
        
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else request.url
        
        if len(content) < 50:
            raise HTTPException(status_code=400, detail="Could not extract meaningful content from URL")
        
        if len(content) > 50000:
            content = content[:50000] + "..."
        
        vector_store = PgVectorStore(collection_name=request.collection_name)
        await vector_store.initialize()
        
        metadata = {
            "title": title[:200],
            "category": request.category,
            "tags": request.tags,
            "source": request.url,
            "source_type": "web_scrape",
            "created_at": datetime.now().isoformat()
        }
        
        doc_id = await vector_store.add_document(content=content, metadata=metadata)
        
        if doc_id:
            return {
                "success": True,
                "document_id": doc_id,
                "title": title[:200],
                "content_length": len(content),
                "message": f"Successfully scraped and stored content from {request.url}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store scraped content")
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect to URL: {str(e)}")
    except Exception as e:
        logger.error(f"Scrape URL error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
async def reindex_collection(collection_name: str = "documents"):
    """
    Reindex the vector store collection by re-ingesting documents
    from the default knowledge base directory.
    
    Only replaces documents sourced from the knowledge_base directory.
    User-uploaded, scraped, and manually created documents are preserved.
    """
    try:
        vector_store = PgVectorStore(collection_name=collection_name)
        await vector_store.initialize()

        stats_before = await vector_store.get_collection_stats()
        count_before = stats_before.get("total_documents", 0) if stats_before else 0

        # (those have source metadata matching the kb path pattern)
        all_docs = await vector_store.list_documents(limit=10000)
        kb_doc_ids = []
        preserved_count = 0
        for doc in all_docs:
            meta = doc.get("metadata", {})
            source = meta.get("source", "")
            file_path = meta.get("file_path", "")
            # KB-ingested docs have source like "blog_posts/file.md" or file_path containing "knowledge_base"
            if "knowledge_base" in file_path or (source and not source.startswith(("http", "manual", "upload", "web"))):
                kb_doc_ids.append(doc.get("id"))
            else:
                preserved_count += 1

        deleted = 0
        for doc_id in kb_doc_ids:
            if doc_id:
                success = await vector_store.delete_document(doc_id)
                if success:
                    deleted += 1

        kb_path = "data/knowledge_base"
        results = await ingest_documents_from_directory(
            directory_path=kb_path,
            collection_name=collection_name,
            validate=True
        )

        final_stats = await vector_store.get_collection_stats()
        count_after = final_stats.get("total_documents", 0) if final_stats else 0

        return {
            "success": results.get("success", False),
            "documents_before": count_before,
            "documents_after": count_after,
            "kb_deleted": deleted,
            "kb_reingested": results.get("documents_ingested", 0),
            "preserved": preserved_count,
            "errors": results.get("errors", []),
            "message": (
                f"Reindexed: removed {deleted} KB docs, re-ingested {results.get('documents_ingested', 0)}. "
                f"Preserved {preserved_count} user-uploaded/scraped docs. Total: {count_before} → {count_after}"
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reindex error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
