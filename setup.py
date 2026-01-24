import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from setuptools import setup, find_packages

setup(
    name="graph-rag-agent",
    version="0.1.0",
    packages=find_packages(),
)