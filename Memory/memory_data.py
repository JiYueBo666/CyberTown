from abc import ABC
from pydantic import BaseModel
from typing import List, Dict, Any
from abc import abstractmethod
from datetime import datetime, timedelta
import heapq


class MemoryItem(BaseModel):
    content: str
    user_id: str
    timestamp: datetime
    content_role: str  # 是用户还是assistant
    importance: float = 0.5


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


class Working(BaseMemory):
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
