from abc import ABC
from pydantic import BaseModel
from typing import List, Dict, Any
from abc import abstractmethod
from datetime import datetime, timedelta
import heapq
import jieba, os
from hello_agents.tools import MemoryTool


class MemoryItem(BaseModel):
    """记忆项数据结构"""

    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}


class MemoryConfig:
    storage_path: str = "./memory_data"

    # 统计显示用的基础配置（仅用于展示）
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作记忆特定配置
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120


class Episode:
    """情景记忆中的单个情景"""

    def __init__(
        self,
        episode_id: str,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        content: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        importance: float = 0.5,
    ):
        self.episode_id = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance


class BaseMemory(ABC):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend

    @abstractmethod
    def add(self, memory_item: MemoryItem):
        pass

    @abstractmethod
    def retrieve(self, query: str, limit=5, **kwargs) -> List[MemoryItem]:
        pass

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """更新记忆

        Args:
            memory_id: 记忆ID
            content: 新内容
            importance: 新重要性
            metadata: 新元数据

        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否删除成功
        """
        pass


class WorkingMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens
        # 纯内存TTL（分钟），可通过在 MemoryConfig 上挂载 working_memory_ttl_minutes 覆盖
        self.max_age_minutes = getattr(self.config, "working_memory_ttl_minutes", 120)
        self.current_tokens = 0
        self.session_start = datetime.now()

        # 内存存储（工作记忆不需要持久化）
        self.memories: List[MemoryItem] = []

        # 使用优先级队列管理记忆
        self.memory_heap = []  # (priority, timestamp, memory_item)

    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆"""
        # 过期清理
        self._expire_old_memories()
        # 计算优先级（重要性 + 时间衰减）
        priority = self._calculate_priority(memory_item)

        # 添加到堆中
        heapq.heappush(
            self.memory_heap, (-priority, memory_item.timestamp, memory_item)
        )
        self.memories.append(memory_item)

        # 更新token计数
        self.current_tokens += len(memory_item.content.split())

        # 检查容量限制
        self._enforce_capacity_limits()

        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, user_id: str = None, **kwargs):
        self._expire_old_memories()
        if not self.memories:
            return []
        # 过滤已遗忘的记忆
        active_memories = [
            m for m in self.memories if not m.metadata.get("forgotten", False)
        ]

        # 按用户ID过滤（如果提供）
        filtered_memories = active_memories
        if user_id:
            filtered_memories = [m for m in active_memories if m.user_id == user_id]

        if not filtered_memories:
            return []

        # 尝试语义向量检索（如果有嵌入模型）
        vector_scores = {}
        try:
            # 简单的语义相似度计算（使用TF-IDF或其他轻量级方法）
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            documents = [query] + [m.content for m in filtered_memories]
            # TF-IDF向量化
            vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)
        except Exception as e:
            # 如果向量检索失败，回退到关键词匹配
            vector_scores = {}

        # 计算最终分数
        query_lower = query.lower()
        scored_memories = []

        for memory in filtered_memories:
            content_lower = memory.content.lower()

            # 获取向量分数（如果有）
            vector_score = vector_scores.get(memory.id, 0.0)
            # 关键词匹配分数
            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / len(content_lower)
            else:
                # 分词匹配
                query_words = set(jieba.cut(query_lower))
                content_words = set(jieba.cut(content_lower))
                intersection = query_words & content_words
                if not intersection:
                    return 0.0

                # 4. Jaccard 相似度 × 权重
                union = query_words | content_words
                jaccard = len(intersection) / len(union)
                keyword_score = jaccard * 0.8
                # 混合分数：向量检索 + 关键词匹配
            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_relevance = keyword_score
            # 时间衰减
            time_decay = self._calculate_time_decay(memory.timestamp)
            base_relevance *= time_decay

            # 重要性权重
            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * importance_weight

            if final_score > 0:
                scored_memories.append((final_score, memory))
        # 按分数排序并返回
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """更新工作记忆"""
        for memory in self.memories:
            if memory.id == memory_id:
                old_tokens = len(memory.content.split())

                if content is not None:
                    memory.content = content
                    # 更新token计数
                    new_tokens = len(content.split())
                    self.current_tokens = self.current_tokens - old_tokens + new_tokens

                if importance is not None:
                    memory.importance = importance

                if metadata is not None:
                    memory.metadata.update(metadata)

                # 重新计算优先级并更新堆
                self._update_heap_priority(memory)

                return True
        return False

    def _expire_old_memories(self):
        """按TTL清理过期记忆，并同步更新堆与token计数"""
        if not self.memories:
            return

        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)
        kept: List[MemoryItem] = []
        removed_token_sum = 0
        for m in self.memories:
            if m.timestamp > cutoff_time:
                kept.append(m)
            else:
                removed_token_sum += len(m.content.split())
        if len(kept) == len(self.memories):
            return
        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_token_sum)

        # 重建堆
        for m in self.memories:
            priority = self._calculate_priority(m)
            heapq.heappush(self.memory_heap, (-priority, m.timestamp, m))

    def _calculate_priority(self, memory_item: MemoryItem):
        priority = memory_item.importance

        time_factor = self._calculate_time_decay(memory_item.timestamp)
        priority *= time_factor
        return priority

    def _calculate_time_decay(self, timestamp: datetime):
        """计算时间衰减因子"""
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600

        # 指数衰减（工作记忆衰减更快）
        decay_factor = self.config.decay_factor ** (hours_passed / 6)  # 每6小时衰减
        return max(0.1, decay_factor)  # 最小保持10%的权重

    def _enforce_capacity_limits(self):
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()
        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory()

    def _remove_lowest_priority_memory(self):
        """删除优先级最低的记忆"""
        if not self.memories:
            return
        # 找到优先级最低的记忆
        lowest_priority = float("inf")
        lowest_memory = None
        for memory in self.memories:
            priority = self._calculate_priority(memory)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_memory = memory
        if lowest_memory:
            self.remove(lowest_memory.id)

    def _update_heap_priority(self, memory: MemoryItem):
        """更新堆中记忆的优先级"""
        # 简单实现：重建堆
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))


class EpisodeMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        # 本地缓存（内存）
        self.episodes: List[Episode] = []
        self.sessions: Dict[str, List[str]] = {}  # session_id -> episode_ids

        # 模式识别缓存
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        # 权威文档存储（SQLite）
        db_dir = (
            self.config.storage_path
            if hasattr(self.config, "storage_path")
            else "./memory_data"
        )
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)

        # 统一嵌入模型（多语言，默认384维）
        self.embedder = get_text_embedder()

        # 向量存储（Qdrant - 使用连接管理器避免重复连接）
        from ..storage.qdrant_store import QdrantConnectionManager

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_store = QdrantConnectionManager.get_instance(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
            vector_size=get_dimension(getattr(self.embedder, "dimension", 384)),
            distance=os.getenv("QDRANT_DISTANCE", "cosine"),
        )

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        # 从元数据中提取情景信息
        session_id = memory_item.metadata.get("session_id", "default_session")
        context = memory_item.metadata.get("context", {})
        outcome = memory_item.metadata.get("outcome")
        participants = memory_item.metadata.get("participants", [])
        tags = memory_item.metadata.get("tags", [])

        # 创建情景（内存缓存）
        episode = Episode(
            episode_id=memory_item.id,
            user_id=memory_item.user_id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
            outcome=outcome,
            importance=memory_item.importance,
        )
        self.episodes.append(episode)
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        # 1) 权威存储（SQLite）
        ts_int = int(memory_item.timestamp.timestamp())
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type="episodic",
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "session_id": session_id,
                "context": context,
                "outcome": outcome,
                "participants": participants,
                "tags": tags,
            },
        )
        # 2) 向量索引（Qdrant）
        try:
            embedding = self.embedder.encode(memory_item.content)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            self.vector_store.add_vectors(
                vectors=[embedding],
                metadata=[
                    {
                        "memory_id": memory_item.id,
                        "user_id": memory_item.user_id,
                        "memory_type": "episodic",
                        "importance": memory_item.importance,
                        "session_id": session_id,
                        "content": memory_item.content,
                    }
                ],
                ids=[memory_item.id],
            )
        except Exception:
            # 向量入库失败不影响权威存储
            pass

        return memory_item.id
