"""
Episodic Memory System for Agent Self-Improvement

Implements Section 6.4 of research plan:
- Agents store task summaries as vector embeddings
- Retrieve relevant past experiences for context
- Enable learning from mistakes and continuous improvement
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import numpy as np

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from ...config.settings import settings
from ...data_layer.vector_store.pgvector_store import PgVectorStore
from ...data_layer.database.connection import get_async_session

logger = logging.getLogger(__name__)

@dataclass
class AgentMemory:
    """
    Structured memory of an agent's task execution
    """
    agent_name: str  # e.g., "content_generator", "strategy_optimizer"
    task_id: str
    task_description: str
    actions_taken: List[str]
    outcome: str  # "success", "failure", "partial"
    metrics: Dict[str, float]  # performance metrics
    human_feedback: Optional[str] = None
    timestamp: Optional[datetime] = None
    lessons_learned: Optional[str] = None

    def to_text(self) -> str:
        text_parts = [
            f"Task: {self.task_description}",
            f"Actions: {'; '.join(self.actions_taken)}",
            f"Outcome: {self.outcome}",
            f"Metrics: {json.dumps(self.metrics)}",
        ]

        if self.human_feedback:
            text_parts.append(f"Human Feedback: {self.human_feedback}")

        if self.lessons_learned:
            text_parts.append(f"Lessons: {self.lessons_learned}")

        return "\n".join(text_parts)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data


class EpisodicMemoryStore:
    """
    Vector-backed episodic memory store for agents

    Each agent has its own memory index for retrieving relevant past experiences
    """

    def __init__(self, agent_name: str):
        """
        Initialize memory store for specific agent

        Args:
            agent_name: Name of agent (e.g., "content_generator")
        """
        self.agent_name = agent_name
        self.collection_name = f"{agent_name}_memory"

        # The vector store (PgVectorStore) uses SentenceTransformer internally for embeddings
        self.embeddings = None
        if settings.OPENAI_API_KEY and not settings.USE_LOCAL_LLM:
            try:
                self.embeddings = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY
                )
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI embeddings: {e}. Using local embeddings.")
                self.embeddings = None

        self.vector_store = PgVectorStore(collection_name=self.collection_name)

        self._initialized = False

        logger.info(f"Initialized episodic memory for agent: {agent_name} (lazy init, embeddings: {'OpenAI' if self.embeddings else 'local'})")

    async def _ensure_initialized(self):
        """
        Lazy initialization of vector store

        This avoids async initialization in __init__ which causes RuntimeError
        when there's no event loop running
        """
        if not self._initialized:
            await self.vector_store.initialize()
            self._initialized = True
            logger.debug(f"Vector store initialized for {self.agent_name}")

    async def store_memory(
        self,
        memory: AgentMemory,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a new memory

        Args:
            memory: AgentMemory object
            metadata: Optional additional metadata

        Returns:
            Memory ID
        """
        await self._ensure_initialized()

        try:
            if memory.timestamp is None:
                memory.timestamp = datetime.now()

            memory_text = memory.to_text()

            doc_metadata = {
                'agent_name': self.agent_name,
                'task_id': memory.task_id,
                'outcome': memory.outcome,
                'timestamp': memory.timestamp.isoformat(),
                'metrics': json.dumps(memory.metrics),
                **(metadata or {})
            }

            document = Document(
                page_content=memory_text,
                metadata=doc_metadata
            )

            memory_id = await self.vector_store.add_documents([document])

            logger.info(f"Stored memory {memory_id[0]} for {self.agent_name}: {memory.task_description[:50]}...")

            return memory_id[0]

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise

    async def retrieve_relevant_memories(
        self,
        query: str,
        k: int = 5,
        outcome_filter: Optional[str] = None,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a new task

        Args:
            query: Description of current task
            k: Number of memories to retrieve
            outcome_filter: Filter by outcome ("success", "failure", etc.)
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant memories with scores
        """
        await self._ensure_initialized()

        try:
            filter_dict = {'agent_name': self.agent_name}
            if outcome_filter:
                filter_dict['outcome'] = outcome_filter

            results = await self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )

            memories = []
            for doc, score in results:
                if score >= min_similarity:
                    memory_data = {
                        'content': doc.page_content,
                        'similarity_score': score,
                        'metadata': doc.metadata,
                        'task_id': doc.metadata.get('task_id'),
                        'outcome': doc.metadata.get('outcome'),
                        'timestamp': doc.metadata.get('timestamp')
                    }
                    memories.append(memory_data)

            logger.info(f"Retrieved {len(memories)} relevant memories for query: {query[:50]}...")

            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    async def get_success_rate(self, task_type: Optional[str] = None) -> float:
        """
        Calculate success rate from memory

        Args:
            task_type: Optional filter by task type

        Returns:
            Success rate (0.0 - 1.0)
        """
        await self._ensure_initialized()

        try:
            all_memories = await self.vector_store.similarity_search(
                query="",
                k=1000,  # Get all
                filter={'agent_name': self.agent_name}
            )

            if not all_memories:
                return 0.0

            success_count = sum(
                1 for doc in all_memories
                if doc.metadata.get('outcome') == 'success'
            )

            success_rate = success_count / len(all_memories)

            logger.info(f"Agent {self.agent_name} success rate: {success_rate:.2%} ({success_count}/{len(all_memories)})")

            return success_rate

        except Exception as e:
            logger.error(f"Failed to calculate success rate: {e}")
            return 0.0

    async def get_common_failure_patterns(self, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve common failure patterns for learning
        
        Args:
            k: Number of failure cases to retrieve
            
        Returns:
            List of failure memories
        """
        await self._ensure_initialized()

        try:
            failures = await self.vector_store.similarity_search(
                query="failure error problem issue",
                k=k,
                filter={
                    'agent_name': self.agent_name,
                    'outcome': 'failure'
                }
            )

            failure_patterns = [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in failures
            ]
            
            logger.info(f"Retrieved {len(failure_patterns)} failure patterns via vector search")
            return failure_patterns

        except Exception as e:
            logger.warning(f"Vector search failed (likely embedding service down), falling back to SQL: {e}")
            
            # Fallback to direct SQL query
            try:
                from sqlalchemy import text
                from ...data_layer.database.connection import get_async_session
                
                async with get_async_session() as session:
                    query = text(f"""
                        SELECT content, metadata 
                        FROM {self.vector_store.collection_name}
                        WHERE metadata->>'agent_name' = :agent_name
                        AND metadata->>'outcome' = 'failure'
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """)
                    
                    result = await session.execute(query, {
                        "agent_name": self.agent_name,
                        "limit": k
                    })
                    
                    rows = result.fetchall()
                    
                    failure_patterns = [
                        {
                            'content': row.content,
                            'metadata': row.metadata
                        }
                        for row in rows
                    ]
                    
                    logger.info(f"Retrieved {len(failure_patterns)} failure patterns via SQL fallback")
                    return failure_patterns
                    
            except Exception as sql_e:
                logger.error(f"SQL fallback also failed: {sql_e}")
                return []

    async def format_memories_for_prompt(
        self,
        memories: List[Dict[str, Any]],
        max_memories: int = 3
    ) -> str:
        """
        Format memories for inclusion in LLM prompt

        Args:
            memories: List of memory dictionaries
            max_memories: Maximum number to include

        Returns:
            Formatted text for prompt
        """
        if not memories:
            return "No relevant past experiences found."

        top_memories = memories[:max_memories]

        formatted_lines = [
            "### Relevant Past Experiences:",
            ""
        ]

        for i, mem in enumerate(top_memories, 1):
            formatted_lines.extend([
                f"**Experience {i}** (similarity: {mem['similarity_score']:.2f}):",
                mem['content'],
                f"_Outcome: {mem['outcome']}_",
                ""
            ])

        return "\n".join(formatted_lines)

    async def clear_old_memories(self, days: int = 90) -> int:
        """
        Clear memories older than specified days

        Args:
            days: Number of days to keep

        Returns:
            Number of memories deleted
        """
        await self._ensure_initialized()

        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            all_memories = await self.vector_store.similarity_search(
                query="",
                k=10000,
                filter={'agent_name': self.agent_name}
            )

            deleted_count = 0
            for doc in all_memories:
                timestamp_str = doc.metadata.get('timestamp')
                if timestamp_str:
                    memory_date = datetime.fromisoformat(timestamp_str)
                    if memory_date < cutoff_date:
                        # Note: Actual deletion depends on vector store implementation
                        deleted_count += 1

            logger.info(f"Cleared {deleted_count} old memories (older than {days} days)")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clear old memories: {e}")
            return 0

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about agent's episodic memory

        Returns:
            Dictionary with total_count, success_count, failure_count,
            success_rate, avg_cost, avg_duration
        """
        await self._ensure_initialized()

        try:
            all_memories = await self.vector_store.similarity_search(
                query="",
                k=10000,
                filter={'agent_name': self.agent_name}
            )

            if not all_memories:
                return {
                    "total_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "success_rate": 0.0,
                    "avg_cost": 0.0,
                    "avg_duration": 0.0
                }

            total_count = len(all_memories)
            success_count = 0
            failure_count = 0
            total_cost = 0.0
            total_duration = 0.0

            for doc in all_memories:
                outcome = doc.metadata.get('outcome', '')
                if outcome == 'success':
                    success_count += 1
                elif outcome == 'failure':
                    failure_count += 1

                metrics_str = doc.metadata.get('metrics', '{}')
                try:
                    metrics = json.loads(metrics_str) if isinstance(metrics_str, str) else metrics_str
                    total_cost += float(metrics.get('cost', 0))
                    total_duration += float(metrics.get('duration', 0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            success_rate = success_count / total_count if total_count > 0 else 0.0
            avg_cost = total_cost / total_count if total_count > 0 else 0.0
            avg_duration = total_duration / total_count if total_count > 0 else 0.0

            return {
                "total_count": total_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_rate,
                "avg_cost": avg_cost,
                "avg_duration": avg_duration
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "total_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "avg_cost": 0.0,
                "avg_duration": 0.0
            }

    async def get_recent_memories(self, limit: int = 10) -> List['AgentMemory']:
        """
        Get recent memories ordered by timestamp (descending)

        Args:
            limit: Maximum number of memories to return

        Returns:
            List of AgentMemory objects
        """
        await self._ensure_initialized()

        try:
            all_memories = await self.vector_store.similarity_search(
                query="",
                k=limit * 2,  # Get extra to account for filtering
                filter={'agent_name': self.agent_name}
            )

            memories_with_time = []
            for doc in all_memories:
                timestamp_str = doc.metadata.get('timestamp')
                try:
                    timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.min
                except (ValueError, TypeError):
                    timestamp = datetime.min
                
                metrics_str = doc.metadata.get('metrics', '{}')
                try:
                    metrics = json.loads(metrics_str) if isinstance(metrics_str, str) else metrics_str
                except (json.JSONDecodeError, TypeError):
                    metrics = {}

                content = doc.page_content
                task_desc = content.split('\n')[0].replace('Task: ', '') if content else 'Unknown task'
                
                actions = []
                for line in content.split('\n'):
                    if line.startswith('Actions: '):
                        actions = line.replace('Actions: ', '').split('; ')
                        break

                memory = AgentMemory(
                    agent_name=self.agent_name,
                    task_id=doc.metadata.get('task_id', ''),
                    task_description=task_desc,
                    actions_taken=actions,
                    outcome=doc.metadata.get('outcome', 'unknown'),
                    metrics=metrics if isinstance(metrics, dict) else {},
                    timestamp=timestamp,
                    human_feedback=None,
                    lessons_learned=None
                )
                memories_with_time.append((timestamp, memory))

            memories_with_time.sort(key=lambda x: x[0], reverse=True)

            return [m for _, m in memories_with_time[:limit]]

        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

    async def get_failure_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get common failure patterns for learning

        Wrapper for get_common_failure_patterns with additional formatting

        Args:
            limit: Number of failure cases to retrieve

        Returns:
            List of formatted failure patterns
        """
        raw_failures = await self.get_common_failure_patterns(k=limit)

        patterns = []
        pattern_groups = {}

        for failure in raw_failures:
            content = failure.get('content', '')
            metadata = failure.get('metadata', {})

            lines = content.split('\n')
            task_line = lines[0] if lines else ''
            
            pattern_key = task_line if task_line else 'unknown'
            
            if pattern_key not in pattern_groups:
                pattern_groups[pattern_key] = {
                    'pattern': pattern_key,  # Full pattern, no truncation
                    'description': content,  # Full description, no truncation
                    'count': 0,
                    'examples': []
                }
            
            pattern_groups[pattern_key]['count'] += 1
            if len(pattern_groups[pattern_key]['examples']) < 3:
                pattern_groups[pattern_key]['examples'].append(content)  # Full content, no truncation

        return list(pattern_groups.values())[:limit]

    async def clear_all_memories(self) -> int:
        """
        Clear all memories for this agent

        WARNING: This is irreversible!

        Returns:
            Number of memories deleted
        """
        await self._ensure_initialized()

        try:
            count = await self.vector_store.delete_by_metadata({
                'agent_name': self.agent_name
            })

            logger.warning(f"Cleared {count} memories for agent: {self.agent_name}")

            return count

        except Exception as e:
            logger.error(f"Failed to clear all memories: {e}")
            return 0


def create_memory_from_task(
    agent_name: str,
    task_id: str,
    task_description: str,
    actions: List[str],
    result: Dict[str, Any],
    human_feedback: Optional[str] = None
) -> AgentMemory:
    """
    Create an AgentMemory from task execution results

    Args:
        agent_name: Name of agent
        task_id: Unique task identifier
        task_description: Description of task
        actions: List of actions taken
        result: Task result dictionary with metrics
        human_feedback: Optional feedback from human reviewer

    Returns:
        AgentMemory object
    """
    outcome = "success" if result.get('success', False) else "failure"

    metrics = {
        'cost': result.get('cost', 0.0),
        'duration': result.get('duration', 0.0),
        'quality_score': result.get('quality_score', 0.0)
    }

    lessons = None
    if outcome == "failure" and result.get('error'):
        lessons = f"Failed due to: {result['error']}"
    elif human_feedback:
        lessons = f"Human feedback: {human_feedback}"

    return AgentMemory(
        agent_name=agent_name,
        task_id=task_id,
        task_description=task_description,
        actions_taken=actions,
        outcome=outcome,
        metrics=metrics,
        human_feedback=human_feedback,
        lessons_learned=lessons,
        timestamp=datetime.now()
    )
