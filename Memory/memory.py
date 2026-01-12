from abc import ABC
from abc import abstractmethod
from Memory.memory_data import MemoryConfig, MemoryItem, BaseMemory
from hello_agents.memory import WorkingMemory, MemoryConfig
from datetime import datetime, timedelta
import heapq


class WorkingMemory(BaseMemory):
    """
    工作记忆实现
    容量有限

    """

    def __init__(self, config: MemoryConfig):
        self.max_capacity = config.working_memory_capacity or 50
        self.max_age_minutes = config.working_memory_ttl_minutes or 60
        self.memories = []
        self.current_tokens = 0
        # 使用优先级队列管理记忆
        self.memory_heap = []  # (priority, timestamp, memory_item)

    def add(self, memory_item: MemoryItem):
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

    def search(self, query: str, limit=5, **kwargs):
        """
        tf-idf检索
        """

        self._expire_old_memories()
        vector_scores = self._try_tfidf_search(query)
        scored_memorie = []

        # 计算综合得分
        for memory in self.memories:
            vector_score = vector_scores.get(memory.id, 0.0)
            keyword_score = self._calculate_keyword_score(query, memory.content)

            base_relevance = (
                vector_score * 0.7 + keyword_score * 0.3
                if vector_score > 0
                else keyword_score
            )

            time_decay = self._calculate_time_decay(memory.timestamp)
            importance_weight = 0.8 + (memory.importance * 0.4)

            final_score = base_relevance * time_decay * importance_weight

            if final_score > 0:
                scored_memorie.append((final_score, memory))
        scored_memorie.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memorie[:limit]]

    def _expire_old_memories(self):
        if not self.memories:
            return

        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)

        kept: list[MemoryItem] = []

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
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))

    def _calculate_priority(self, memory: MemoryItem) -> float:
        """计算记忆优先级"""
        # 基础优先级 = 重要性
        priority = memory.importance

        # 时间衰减
        time_decay = self._calculate_time_decay(memory.timestamp)
        priority *= time_decay

        return priority

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """计算时间衰减因子"""
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600

        # 指数衰减（工作记忆衰减更快）
        decay_factor = self.config.decay_factor ** (hours_passed / 6)  # 每6小时衰减
        return max(0.1, decay_factor)  # 最小保持10%的权重

    def _enforce_capacity_limits(self):
        """强制执行容量限制"""
        # 检查记忆数量限制
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()

        # 检查token限制
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
