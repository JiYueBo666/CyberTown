from typing import List, Any, Dict
from datetime import datetime

from Tool.base import Tool, ToolParameter
from Memory.memory_manager import MemoryManager, MemoryConfig


class MemoryTool(Tool):
    """记忆工具
    为Agent提供记忆功能：
    - 添加记忆
    - 检索相关记忆
    - 获取记忆摘要
    - 管理记忆生命周期
    """

    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: MemoryConfig = None,
        memory_types: List[str] = None,
    ):
        super().__init__(
            name="memory", description="记忆工具 - 可以存储和检索对话历史、知识和经验"
        )
        self.memory_config = memory_config or MemoryConfig()
        self.memory_types = memory_types or ["working", "episodic", "semantic"]
        self.memory_manager = MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working="working" in self.memory_types,  # True为启用
            enable_episodic="episodic" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
            enable_perceptual="perceptual" in self.memory_types,
        )
        self.current_session_id = None
        self.conversation_count = 0

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:

        if not self.validate_parameters(parameters):
            raise ValueError("参数验证失败")
        action = parameters.get("action")
        kwargs = {k: v for k, v in parameters.items() if k != "action"}

        return self.execute(action, **kwargs)

    def execute(self, action: str, **kwargs):
        """
        执行操作，
        add:添加
        search:检索记忆
        summary:获取记忆摘要
        stats: 获取记忆统计信息
        """
        if action == "add":
            return self._add_memory(**kwargs)
        elif action == "search":
            return self._search_memory(**kwargs)
        elif action == "summary":
            return self._get_summary(**kwargs)
        elif action == "stats":
            return self._get_stats()
        elif action == "update":
            return self._update_memory(**kwargs)
        elif action == "remove":
            return self._remove_memory(**kwargs)
        elif action == "forget":
            return self._forget(**kwargs)
        elif action == "consolidate":
            return self._consolidate(**kwargs)
        elif action == "clear_all":
            return self._clear_all()
        else:
            return f"不支持的操作: {action}。支持的操作: add, search, summary, stats, update, remove, forget, consolidate, clear_all"

    def _add_memory(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        file_path: str = None,
        modality: str = None,
        **metadata,
    ) -> str:
        try:
            # 确保会话ID存在
            if self.current_session_id is None:
                self.current_session_id = (
                    f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            metadata.update(
                {
                    "session_id": self.current_session_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata,
                auto_classify=False,  # 禁用自动分类，使用明确指定的类型
            )

            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"

        except Exception as e:
            return f"❌ 添加记忆失败: {str(e)}"

    def _search_memory(
        self,
        query: str,
        limit: int = 5,
        memory_types: List[str] = None,
        memory_type: str = None,  # 添加单数形式的参数支持
        min_importance: float = 0.1,
    ) -> str:
        """搜索记忆"""
        try:
            # 处理单数形式的memory_type参数
            if memory_type and not memory_types:
                memory_types = [memory_type]

            results = self.memory_manager.retrieve_memories(
                query=query,
                limit=limit,
                memory_types=memory_types,
                min_importance=min_importance,
            )

            if not results:
                return f"🔍 未找到与 '{query}' 相关的记忆"

            # 格式化结果
            formatted_results = []
            formatted_results.append(f"🔍 找到 {len(results)} 条相关记忆:")

            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆",
                }.get(memory.memory_type, memory.memory_type)

                content_preview = (
                    memory.content[:80] + "..."
                    if len(memory.content) > 80
                    else memory.content
                )
                formatted_results.append(
                    f"{i}. [{memory_type_label}] {content_preview} (重要性: {memory.importance:.2f})"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            return f"❌ 搜索记忆失败: {str(e)}"

    def _get_summary(self, limit: int = 10) -> str:
        """获取记忆摘要"""
        try:
            stats = self.memory_manager.get_memory_stats()

            summary_parts = [
                f"📊 记忆系统摘要",
                f"总记忆数: {stats['total_memories']}",
                f"当前会话: {self.current_session_id or '未开始'}",
                f"对话轮次: {self.conversation_count}",
            ]

            # 各类型记忆统计
            if stats["memories_by_type"]:
                summary_parts.append("\n📋 记忆类型分布:")
                for memory_type, type_stats in stats["memories_by_type"].items():
                    count = type_stats.get("count", 0)
                    avg_importance = type_stats.get("avg_importance", 0)
                    type_label = {
                        "working": "工作记忆",
                        "episodic": "情景记忆",
                        "semantic": "语义记忆",
                        "perceptual": "感知记忆",
                    }.get(memory_type, memory_type)

                    summary_parts.append(
                        f"  • {type_label}: {count} 条 (平均重要性: {avg_importance:.2f})"
                    )

            # 获取重要记忆 - 修复重复问题
            important_memories = self.memory_manager.retrieve_memories(
                query="",
                memory_types=None,  # 从所有类型中检索
                limit=limit * 3,  # 获取更多候选，然后去重
                min_importance=0.5,  # 降低阈值以获取更多记忆
            )

            if important_memories:
                # 去重：使用记忆ID和内容双重去重
                seen_ids = set()
                seen_contents = set()
                unique_memories = []

                for memory in important_memories:
                    # 使用ID去重
                    if memory.id in seen_ids:
                        continue

                    # 使用内容去重（防止相同内容的不同记忆）
                    content_key = memory.content.strip().lower()
                    if content_key in seen_contents:
                        continue

                    seen_ids.add(memory.id)
                    seen_contents.add(content_key)
                    unique_memories.append(memory)

                # 按重要性排序
                unique_memories.sort(key=lambda x: x.importance, reverse=True)
                summary_parts.append(
                    f"\n⭐ 重要记忆 (前{min(limit, len(unique_memories))}条):"
                )

                for i, memory in enumerate(unique_memories[:limit], 1):
                    content_preview = (
                        memory.content[:60] + "..."
                        if len(memory.content) > 60
                        else memory.content
                    )
                    summary_parts.append(
                        f"  {i}. {content_preview} (重要性: {memory.importance:.2f})"
                    )

            return "\n".join(summary_parts)

        except Exception as e:
            return f"❌ 获取摘要失败: {str(e)}"

    def _get_stats(self) -> str:
        """获取统计信息"""
        try:
            stats = self.memory_manager.get_memory_stats()

            stats_info = [
                f"📈 记忆系统统计",
                f"总记忆数: {stats['total_memories']}",
                f"启用的记忆类型: {', '.join(stats['enabled_types'])}",
                f"会话ID: {self.current_session_id or '未开始'}",
                f"对话轮次: {self.conversation_count}",
            ]

            return "\n".join(stats_info)

        except Exception as e:
            return f"❌ 获取统计信息失败: {str(e)}"
