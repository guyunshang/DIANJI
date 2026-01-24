import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from abc import ABC, abstractmethod


class CacheKeyStrategy(ABC):
    """缓存键生成策略的抽象基类"""
    
    @abstractmethod
    def generate_key(self, query: str, **kwargs) -> str:
        """生成缓存键"""
        pass