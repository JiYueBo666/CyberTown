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
        )
        self.current_session_id = None
        self.conversation_count = 0

    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool的统一抽象接口
        需要进行参数验证
        """

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
        """
        tool层，确保会话状态。调用manager层的add
        更新信息元数据，包括会话ID和当前时间戳
        """
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
        """
        搜索记忆
        搜索指定列表中类型记忆，并进行格式化返回
        """
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
        """获取记忆摘要，格式化输出"""
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

    def auto_record_conversation(self, user_input: str, agent_response: str):
        """自动记录对话

        这个方法可以被Agent调用来自动记录对话历史

        根据关键字决定是否转为长期记忆
        """
        self.conversation_count += 1
        # 记录用户输入
        self._add_memory(
            content=f"用户: {user_input}",
            memory_type="working",
            importance=0.6,
            type="user_input",
            conversation_id=self.conversation_count,
        )

        # 记录Agent响应
        self._add_memory(
            content=f"助手: {agent_response}",
            memory_type="working",
            importance=0.7,
            type="agent_response",
            conversation_id=self.conversation_count,
        )

        # 如果是重要对话，记录为情景记忆
        if len(agent_response) > 100 or "重要" in user_input or "记住" in user_input:
            interaction_content = f"对话 - 用户: {user_input}\n助手: {agent_response}"
            self._add_memory(
                content=interaction_content,
                memory_type="episodic",
                importance=0.8,
                type="interaction",
                conversation_id=self.conversation_count,
            )

    def _update_memory(
        self, memory_id: str, content: str = None, importance: float = None, **metadata
    ):
        """更新记忆"""
        try:
            success = self.memory_manager.update_memory(
                memory_id=memory_id,
                content=content,
                importance=importance,
                metadata=metadata or None,
            )
            return "✅ 记忆已更新" if success else "⚠️ 未找到要更新的记忆"
        except Exception as e:
            return f"❌ 更新记忆失败: {str(e)}"

    def _remove_memory(self, memory_id):
        """删除记忆"""
        try:
            success = self.memory_manager.remove_memory(memory_id)
            return "✅ 记忆已删除" if success else "⚠️ 未找到要删除的记忆"
        except Exception as e:
            return f"❌ 删除记忆失败: {str(e)}"

    def _forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ):
        """遗忘记忆（支持多种策略）"""
        try:
            count = self.memory_manager.forget_memories(
                strategy=strategy, threshold=threshold, max_age_days=max_age_days
            )
            return f"🧹 已遗忘 {count} 条记忆（策略: {strategy}）"
        except Exception as e:
            return f"❌ 遗忘记忆失败: {str(e)}"

    def _consolidate(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
    ):
        """整合记忆（将重要的短期记忆提升为长期记忆）"""
        try:
            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold,
            )
            return f"🔄 已整合 {count} 条记忆为长期记忆（{from_type} → {to_type}，阈值={importance_threshold}）"
        except Exception as e:
            return f"❌ 整合记忆失败: {str(e)}"

    def _clear_all(self):
        """清空所有记忆"""
        try:
            self.memory_manager.clear_all_memories()
            return "🧽 已清空所有记忆"
        except Exception as e:
            return f"❌ 清空记忆失败: {str(e)}"

    def get_context_for_query(self, query: str, limit: int = 3) -> str:
        """为查询获取相关上下文

        这个方法可以被Agent调用来获取相关的记忆上下文
        """
        results = self.memory_manager.retrieve_memories(
            query=query, limit=limit, min_importance=0.3
        )

        if not results:
            return ""

        context_parts = ["相关记忆:"]
        for memory in results:
            context_parts.append(f"- {memory.content}")

        return "\n".join(context_parts)

    def clear_session(self):
        """清除当前会话"""
        self.current_session_id = None
        self.conversation_count = 0

        # 清理工作记忆
        wm = (
            self.memory_manager.memory_types.get("working")
            if hasattr(self.memory_manager, "memory_types")
            else None
        )
        if wm:
            wm.clear()

    def consolidate_memories(self):
        """整合记忆"""
        return self.memory_manager.consolidate_memories()

    def forget_old_memories(self, max_age_days: int = 30):
        """遗忘旧记忆"""
        return self.memory_manager.forget_memories(
            strategy="time_based", max_age_days=max_age_days
        )
