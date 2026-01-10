import yaml
from typing import List, Dict, Any
from utils import load_npc_info
from agent.npc_agent import create_npc_base_info, NPCAgent
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    npc_info = load_npc_info("./configs/npc.yaml")
    from llm import AgentLLM

    llm = AgentLLM()

    npc_list = []
    npc_dict = {}  # 用于通过索引快速查找
    if npc_info:
        for info in npc_info:
            npc_system_prompt = create_npc_base_info(
                npc_id=info["id"],
                name=info["name"],
                persona=info["persona"],
                profession=info["profession"],
                llm=llm,
            )
            agent = NPCAgent(
                name=info["name"],
                llm=llm,
                system_prompt=npc_system_prompt,
            )
            npc_list.append(agent)
            npc_dict[info["id"]] = agent  # 可选，但建议保留 id 映射
            print(f"--npc {info['name']}创建成功---")
    else:
        print("❌ 未加载到任何 NPC，请检查 ./configs/npc.yaml")
        exit(1)

    # ========== 主交互循环 ==========
    while True:
        print("\n" + "=" * 40)
        print("欢迎来到赛博小镇！")
        print("当前可互动的 NPC：")
        for i, agent in enumerate(npc_list, 1):
            # 从 agent.name 反推 profession？或你需要额外存 profession
            # 这里假设你可以在 agent 中访问 profession，或从 npc_info 获取
            # 简单起见，我们用 npc_info[i-1]["profession"]
            print(f"[{i}] {agent.name}")
        print("[q] 退出小镇")
        print("=" * 40)

        choice = input("请选择 NPC 编号（输入 q 退出）: ").strip()

        if choice.lower() == "q":
            print("👋 欢迎下次再来！")
            break

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(npc_list):
                raise ValueError
            selected_agent = npc_list[idx]
            name = selected_agent.name
        except (ValueError, IndexError):
            print("⚠️ 无效选择，请输入有效编号。")
            continue

        # 进入与该 NPC 的对话
        print(f"\n💬 正在与 {name} 对话中...（输入 quit 返回主菜单）")
        while True:
            user_input = input("> ").strip()
            if user_input.lower() == "quit":
                print(f"🔚 结束与 {name} 的对话。\n")
                break
            if not user_input:
                continue

            try:
                # 调用你的 Agent 对话方法
                response = selected_agent.run(user_input)  # ← 替换为你的实际方法名
            except Exception as e:
                print(f"❌ 对话出错: {e}")
