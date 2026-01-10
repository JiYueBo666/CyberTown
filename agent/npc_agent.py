from llm import AgentLLM


class NPCAgent:
    def __init__(self, name, llm: AgentLLM, system_prompt):
        self.llm = llm
        self.system_prompt = system_prompt
        self.name = name

    def run(self, user_input):
        return self._chat_stream(user_input)

    def _chat_stream(self, user_message: str, temperature: float = 0.2) -> str:
        """
        Agent 层控制流式输出：决定“何时打印名字”、“如何逐字显示”
        实现了解耦：LLM 只给 token，Agent 决定怎么 show
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        print(f"{self.name}：", end="", flush=True)

        full_reply = []
        try:
            for token in self.llm.stream_gengrate(messages, temperature):
                print(token, end="", flush=True)
                full_reply.append(token)
        except Exception as e:
            error_msg = f"\n⚠️ {str(e)}"
            print(error_msg)
            return error_msg

        print()  # 最后换行
        return "".join(full_reply)

    def _chat(self, user_input):
        message = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        return self.llm.think(message)


def create_npc_base_info(npc_id: str, name: str, persona: str, profession: str, llm):
    # 构建npc系统提示词
    system_prompt = f"""
        你是{name},一名{profession}.
        你的性格特点是:{persona}.

        你在赛博小镇生活着。

        请你根据你的角色和性格，自然的与玩家对话交流。
        请记住你们之前的对话内容，保持对话连贯性。
    """

    return system_prompt
