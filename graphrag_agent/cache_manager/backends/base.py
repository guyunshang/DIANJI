import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from abc import ABC, abstractmethod
from typing import Any, Optional


class CacheStorageBackend(ABC):
    """缓存存储后端的抽象基类"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """设置缓存项"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass