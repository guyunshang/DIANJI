import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from .settings import (
    SERVER_HOST,
    SERVER_PORT,
    SERVER_RELOAD,
    SERVER_LOG_LEVEL,
    SERVER_WORKERS,
    UVICORN_CONFIG,
)
