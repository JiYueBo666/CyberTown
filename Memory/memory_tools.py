from hello_agents.memory import MemoryManager, MemoryItem


class MemoryTool:
    def __init__(self, memory_config, memory_types):
        pass

    def execute(self, action: str, **kwargs):
        if action == "add":
            return self._add_memory(**kwargs)
        elif action == "search":
            return self._search_memory(**kwargs)
        elif action == "summary":
            return self._get_memory(**kwargs)

    def __add_memory(
        self,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        file_path: str = None,
        modality: str = None,
    ):
        pass
