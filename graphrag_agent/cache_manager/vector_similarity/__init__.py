import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from .matcher import VectorSimilarityMatcher
from .embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbedding,
    OpenAIEmbeddingProvider,
    get_cache_embedding_provider
)

__all__ = [
    'VectorSimilarityMatcher',
    'EmbeddingProvider',
    'SentenceTransformerEmbedding',
    'OpenAIEmbeddingProvider',
    'get_cache_embedding_provider'
]