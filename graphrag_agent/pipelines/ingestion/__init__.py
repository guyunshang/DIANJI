import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from graphrag_agent.pipelines.ingestion.document_processor import DocumentProcessor
from graphrag_agent.pipelines.ingestion.file_reader import FileReader
from graphrag_agent.pipelines.ingestion.text_chunker import ChineseTextChunker

__all__ = [
    'DocumentProcessor',
    'FileReader',
    'ChineseTextChunker'
]
