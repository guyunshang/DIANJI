import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from graphrag_agent.graph.core import (
    GraphConnectionManager, 
    connection_manager,
    BaseIndexer,
    timer,
    generate_hash,
    batch_process,
    retry,
    get_performance_stats,
    print_performance_stats
)

# Indexing
from graphrag_agent.graph.indexing import (
    ChunkIndexManager,
    EntityIndexManager
)

# Structure
from graphrag_agent.graph.structure import (
    GraphStructureBuilder
)

# Extraction
from graphrag_agent.graph.extraction import (
    EntityRelationExtractor,
    GraphWriter
)

# Similar Entity
from graphrag_agent.graph.processing import (
    EntityMerger,
    SimilarEntityDetector,
    GDSConfig,
    EntityDisambiguator,
    EntityAligner,
    EntityQualityProcessor
)

__all__ = [
    # Core
    'GraphConnectionManager',
    'connection_manager',
    'BaseIndexer',
    'timer',
    'generate_hash',
    'batch_process',
    'retry',
    'get_performance_stats',
    'print_performance_stats',
    
    # Indexing
    'ChunkIndexManager',
    'EntityIndexManager',
    
    # Structure
    'GraphStructureBuilder',
    
    # Extraction
    'EntityRelationExtractor',
    'GraphWriter',
    
    # Processing
    'EntityMerger',
    'SimilarEntityDetector',
    'GDSConfig',
    'EntityDisambiguator',
    'EntityAligner',
    'EntityQualityProcessor'
]