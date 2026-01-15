from .memory_data import MemoryConfig, MemoryItem, WorkingMemory
import os
from typing import Dict, DefaultDict, Any, List, Optional
import jieba
import json, datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器 - 统一的记忆操作接口

    负责：
    - 记忆生命周期管理
    - 记忆优先级和重要性评估
    - 记忆遗忘和清理机制
    - 多类型记忆的协调管理
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id
        # 初始化各类型记忆
        self.memory_types = {}

        if enable_working:
            self.memory_types["working"] = WorkingMemory(self.config)

        if enable_episodic:
            self.memory_types["episodic"] = EpisodicMemory(self.config)

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_classify: bool = True,
    ) -> str:
        """
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性分数 (0-1)
            metadata: 元数据
            auto_classify: 是否自动分类到合适的记忆类型
        Returns:
            记忆ID
        """

        # 创建记忆项
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {},
        )

        if memory_type in self.memory_types:
            memory_id = self.memory_types[memory_type].add(memory_item)
            return memory_id
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")

    def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        time_range: Optional[tuple] = None,
    ) -> List[MemoryItem]:
        """检索记忆

        Args:
            query: 查询内容
            memory_types: 要检索的记忆类型列表
            limit: 返回数量限制
            min_importance: 最小重要性阈值
            time_range: 时间范围 (start_time, end_time)

        Returns:
            检索到的记忆列表
        """
        if memory_types is None:
            memory_types = list(self.memory_types.keys())
        # 从各个记忆类型中检索
        all_results = []
        per_type_limit = max(1, limit // len(memory_types))

        for memory_type in memory_types:
            if memory_type in self.memory_types:
                memory_instance = self.memory_types[memory_type]
                try:
                    # 使用各个记忆类型自己的检索方法
                    type_results = memory_instance.retrieve(
                        query=query,
                        limit=per_type_limit,
                        min_importance=min_importance,
                        user_id=self.user_id,
                    )
                    all_results.extend(type_results)
                except Exception as e:
                    logger.warning(f"检索 {memory_type} 记忆时出错: {e}")
                    continue
        all_results.sort(key=lambda x: x.importance, reverse=True)
        return all_results[:limit]
