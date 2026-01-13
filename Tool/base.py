from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Dict


class ToolParameter(BaseModel):
    """工具参数定义"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> Any:
        """
        执行工具
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, ToolParameter]:
        """
        获取工具参数
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.dict() for param in self.get_parameters()],
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
