import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

"""在多智能体命名空间下转出检索适配工具函数。"""

from graphrag_agent.search.retrieval_adapter import (
    create_retrieval_metadata,
    create_retrieval_result,
    merge_retrieval_results,
    results_from_documents,
    results_from_entities,
    results_from_relationships,
    results_to_payload,
)

__all__ = [
    "create_retrieval_metadata",
    "create_retrieval_result",
    "merge_retrieval_results",
    "results_from_documents",
    "results_from_entities",
    "results_from_relationships",
    "results_to_payload",
]
