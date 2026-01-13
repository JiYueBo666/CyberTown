from .memory_data import MemoryConfig, MemoryItem
import os
from typing import Dict, DefaultDict, Any, List, Optional
import jieba
import json, datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import logging


class MemoryManager:
    def __init__(self, memory_config: MemoryConfig, max_envent_num=15):

        # npc记忆
        self.npc_memories = Dict[str, List[MemoryItem]]
        self.memory_path = MemoryConfig.storage_path

        self.max_event_num = max_envent_num or 15  # 最大事件容量

        self.vectorize = TfidfVectorizer(
            tokenizer=self._tokenize, lowercase=True, stop_words=""
        )
        self.load_npc_memory()

    def execute(self, action: str, **kwargs):
        pass

    def _add_event(self, event: MemoryItem):
        if event.user_id not in self.npc_memories:
            self.npc_memories[event.user_id] = [event]
        else:
            self.npc_memories[event.user_id].append(event)
        self._clear_out_memory(npc_name=event.user_id)

    def retrieve_memory(self, query, user_id, top_k=5, alpha=0.7):
        """
        tf-idf search
        """

        if not self.npc_memories:
            return []
        corpus = [
            self.npc_memories[user_id][i].content
            for i in range(len(self.npc_memories[user_id]))
        ]

        try:
            tfidf_matrix = self.vectorize.fit_transform(corpus)
            query_vec = self.vectorize.transform([query])
        except Exception as e:
            return self.npc_memories[user_id][-top_k:]

        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        scores = []

        for i, mem in enumerate(self.npc_memories[user_id]):
            sim_score = float(similarity[i])
            final_score = alpha * sim_score + (1 - alpha) * mem.importance
            scores.append((final_score, mem))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scores[:top_k]]

    def load_npc_memory(self):
        """从单个 JSON 文件加载所有 NPC 记忆（兼容低Python版本）"""
        if not os.path.exists(self.memory_path):
            self.npc_memories = {}
            print(f"⚠️ 记忆文件不存在：{self.memory_path}")
            return

        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            loaded_memories = {}
            for user_id, mem_list in raw_data.items():
                loaded_memories[user_id] = []
                for idx, mem in enumerate(mem_list):
                    try:
                        # 1. 校验必填字段
                        required_fields = [
                            "content",
                            "user_id",
                            "timestamp",
                            "content_role",
                        ]
                        for field in required_fields:
                            if field not in mem:
                                raise ValueError(f"缺失必填字段：{field}")

                        # 2. 兼容所有Python版本的时间解析（替换fromisoformat）
                        # 适配格式：2026-01-12T11:08:02.401034
                        timestamp_str = mem["timestamp"]
                        try:
                            # 核心修改：用strptime替代fromisoformat
                            timestamp = datetime.datetime.strptime(
                                timestamp_str, "%Y-%m-%dT%H:%M:%S.%f"
                            )
                        except ValueError as e:
                            # 兼容无微秒的情况（比如 2026-01-12T11:08:02）
                            timestamp = datetime.datetime.strptime(
                                timestamp_str, "%Y-%m-%dT%H:%M:%S"
                            )

                        # 3. 初始化MemoryItem
                        memory_item = MemoryItem(
                            content=mem["content"],
                            user_id=mem["user_id"],
                            timestamp=timestamp,
                            content_role=mem["content_role"],
                            importance=mem.get("importance", 0.5),
                        )
                        loaded_memories[user_id].append(memory_item)

                    except Exception as sub_e:
                        print(
                            f"❌ 解析第{idx+1}条记忆失败（user_id={user_id}）：{sub_e}"
                        )
                        print(f"   出错数据：{mem}")
                        continue

            self.npc_memories = loaded_memories
            print(f"✅ 成功加载 {len(self.npc_memories)} 个 NPC 的记忆")

        except Exception as e:
            print(f"⚠️ 加载记忆文件失败: {e}")
            self.npc_memories = {}

    def _save_npc_memory(self):
        """仅更新内存中的记忆，并自动裁剪 + 持久化到文件"""
        # 自动裁剪（保留最近 max_event_num 条）
        for npc_name in self.npc_memories:
            if len(self.npc_memories[npc_name]) > self.max_event_num:
                self.npc_memories[npc_name] = self.npc_memories[npc_name][
                    -self.max_event_num :
                ]

        # 立即保存到磁盘（简单策略）
        self._persist_memories()

    def _persist_memories(self):
        """将整个 self.npc_memories 写入单个 JSON 文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)

        # 转为可序列化的 dict
        serializable = {}
        for user_id, mem_list in self.npc_memories.items():
            serializable[user_id] = [
                {
                    "content": m.content,
                    "user_id": m.user_id,
                    "timestamp": m.timestamp.isoformat(),
                    "importance": m.importance,
                    "content_role": m.content_role,
                }
                for m in mem_list
            ]

        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存记忆失败: {e}")

    def _clear_out_memory(self, npc_name: str):
        if len(self.npc_memories[npc_name]) > self.max_event_num:
            self.npc_memories[npc_name] = self.npc_memories[npc_name][
                -self.max_event_num :
            ]  # 删除最前面的记忆

    def _tokenize(self, text: str):
        # 简单判断是否含中文
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return list(jieba.cut(text))
        else:
            return text.split()


class MemoryManager2:
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
        enable_semantic: bool = True,
        enable_perceptual: bool = False,
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id
        # 初始化各类型记忆
        self.memory_types = {}

        if enable_working:
            self.memory_types["working"] = WorkingMemory(self.config)

        if enable_episodic:
            self.memory_types["episodic"] = EpisodicMemory(self.config)

        if enable_semantic:
            self.memory_types["semantic"] = SemanticMemory(self.config)

        if enable_perceptual:
            self.memory_types["perceptual"] = PerceptualMemory(self.config)

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
