import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

import hashlib
from .base import CacheKeyStrategy


class SimpleCacheKeyStrategy(CacheKeyStrategy):
    """简单的MD5哈希缓存键策略"""
    
    def generate_key(self, query: str, **kwargs) -> str:
        """使用查询字符串的MD5哈希生成缓存键"""
        return hashlib.md5(query.strip().encode('utf-8')).hexdigest()