import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from graphrag_agent.evaluation.preprocessing.text_cleaner import clean_references, clean_thinking_process
from graphrag_agent.evaluation.preprocessing.reference_extractor import extract_references_from_answer

__all__ = [
    'clean_references',
    'clean_thinking_process',
    'extract_references_from_answer'
]