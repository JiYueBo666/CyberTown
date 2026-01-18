from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
import threading


class DocumentStore(ABC):
    @abstractmethod
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Dict[str, Any] = None,
    ):
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单个记忆"""
        pass

    @abstractmethod
    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """搜索记忆"""
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        properties: Dict[str, Any] = None,
    ) -> bool:
        """更新记忆"""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        pass

    @abstractmethod
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加文档"""
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        pass
