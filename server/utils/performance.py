import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

import time
import functools


def measure_performance(endpoint_name):
    """
    性能测量装饰器
    
    Args:
        endpoint_name: API端点名称
        
    Returns:
        装饰后的函数
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # 记录性能
                duration = time.time() - start_time
                print(f"API性能 - {endpoint_name}: {duration:.4f}s")
                
                return result
            except Exception as e:
                # 记录异常和性能
                duration = time.time() - start_time
                print(f"API异常 - {endpoint_name}: {str(e)} ({duration:.4f}s)")
                raise
                
        return wrapper
    return decorator