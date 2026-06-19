import logging
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import numpy as np

from .pgvector_store import PgVectorStore
from ...config.settings import settings

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Semantic cache for LLM responses
    Reduces API costs by caching similar prompts
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_hours: int = 24
    ):
        self.vector_store = PgVectorStore(collection_name="semantic_cache")
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
        self._embedding_model = None
        self.enabled = settings.ENABLE_SEMANTIC_CACHE

    @property
    def embedding_model(self):
        """Lazy-load SentenceTransformer to avoid crashing startup if HuggingFace is unreachable."""
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                raise
        return self._embedding_model
    
    async def initialize(self):
        """Initialize cache"""
        await self.vector_store.initialize()
        logger.info("Semantic cache initialized")
    
    def _create_cache_key(self, prompt: str, model: str) -> str:
        """
        Create unique cache key
        
        Args:
            prompt: Input prompt
            model: Model name
        
        Returns:
            Cache key
        """
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get(
        self,
        prompt: str,
        model: str = "gpt-4",
        campaign_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response
        
        Args:
            prompt: Input prompt
            model: Model name
            campaign_id: Optional campaign ID to scope cache per-campaign
        
        Returns:
            Cached response or None
        """
        if not self.enabled:
            return None
        
        try:
            filter_meta = {"model": model}
            if campaign_id:
                filter_meta["campaign_id"] = campaign_id

            results = await self.vector_store.search(
                query=prompt,
                top_k=1,
                filter_metadata=filter_meta
            )
            
            if not results:
                logger.debug("Cache miss")
                return None
            
            result = results[0]
            similarity = result['similarity']
            
            if similarity < self.similarity_threshold:
                logger.debug(f"Similarity too low: {similarity}")
                return None
            
            cached_at = datetime.fromisoformat(
                result['metadata'].get('cached_at')
            )
            age = datetime.utcnow() - cached_at
            
            if age > timedelta(hours=self.ttl_hours):
                logger.debug("Cache entry expired")
                await self.vector_store.delete_document(result['id'])
                return None
            
            logger.info(f"Cache hit (similarity: {similarity:.4f})")
            
            return {
                "response": result['metadata'].get('response'),
                "model": result['metadata'].get('model'),
                "cached_at": result['metadata'].get('cached_at'),
                "similarity": similarity
            }
        
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    async def set(
        self,
        prompt: str,
        response: str,
        model: str = "gpt-4",
        metadata: Optional[Dict[str, Any]] = None,
        campaign_id: Optional[str] = None
    ) -> bool:
        """
        Cache response
        
        Args:
            prompt: Input prompt
            response: Model response
            model: Model name
            metadata: Additional metadata
            campaign_id: Optional campaign ID to scope cache per-campaign
        
        Returns:
            Success boolean
        """
        if not self.enabled:
            return False
        
        try:
            cache_key = self._create_cache_key(prompt, model)
            
            cache_metadata = {
                "cache_key": cache_key,
                "model": model,
                "response": response,
                "cached_at": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            if campaign_id:
                cache_metadata["campaign_id"] = campaign_id
            
            await self.vector_store.add_document(
                content=prompt,
                metadata=cache_metadata
            )
            
            logger.info("Response cached successfully")
            return True
        
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    async def invalidate(self, prompt: str, model: str) -> bool:
        """
        Invalidate cache entry
        
        Args:
            prompt: Input prompt
            model: Model name
        
        Returns:
            Success boolean
        """
        try:
            cache_key = self._create_cache_key(prompt, model)
            
            results = await self.vector_store.search(
                query=prompt,
                top_k=1,
                filter_metadata={"cache_key": cache_key}
            )
            
            if results:
                await self.vector_store.delete_document(results[0]['id'])
                logger.info("Cache entry invalidated")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Cache invalidate failed: {e}")
            return False

    async def invalidate_campaign(self, campaign_id: str) -> int:
        """Invalidate all cached entries for a campaign (e.g. after rejection)."""
        try:
            count = await self.vector_store.delete_by_metadata(
                {"campaign_id": campaign_id}
            )
            if count > 0:
                logger.info(f"Invalidated {count} cache entries for campaign {campaign_id}")
            return count
        except Exception as e:
            logger.error(f"Campaign cache invalidation failed: {e}")
            return 0
    
    async def clear_expired(self) -> int:
        """
        Clear expired cache entries
        
        Returns:
            Number of entries cleared
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours)
            
            stats = await self.vector_store.get_collection_stats()
            
            logger.info(f"Cleared expired cache entries")
            return 0
        
        except Exception as e:
            logger.error(f"Failed to clear expired: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Cache stats
        """
        stats = await self.vector_store.get_collection_stats()
        
        return {
            "enabled": self.enabled,
            "total_entries": stats.get("total_documents", 0),
            "similarity_threshold": self.similarity_threshold,
            "ttl_hours": self.ttl_hours
        }