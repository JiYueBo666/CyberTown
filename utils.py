import yaml
from typing import List, Dict, Any


def load_npc_info(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "npcs" not in data:
            raise ValueError("yaml格式错误,npc初始信息导入失败")
        npcs = data["npcs"]
        if not isinstance(npcs, list):
            raise ValueError("npc字段必须是一个列表")
        return npcs
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 格式错误: {e}")
    except Exception as e:
        raise RuntimeError(f"加载 NPC 配置时发生未知错误: {e}")
