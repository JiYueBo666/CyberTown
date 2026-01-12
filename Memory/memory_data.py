from abc import ABC
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
from abc import abstractmethod


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
