import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from graphrag_agent.search.tool.reasoning.nlp import extract_between, extract_from_templates, extract_sentences
from graphrag_agent.search.tool.reasoning.prompts import kb_prompt, num_tokens_from_string
from graphrag_agent.search.tool.reasoning.thinking import ThinkingEngine
from graphrag_agent.search.tool.reasoning.validator import AnswerValidator
from graphrag_agent.search.tool.reasoning.search import DualPathSearcher, QueryGenerator
from graphrag_agent.search.tool.reasoning.community_enhance import CommunityAwareSearchEnhancer
from graphrag_agent.search.tool.reasoning.kg_builder import DynamicKnowledgeGraphBuilder
from graphrag_agent.search.tool.reasoning.evidence import EvidenceChainTracker

__all__ = [
    "extract_between",
    "extract_from_templates",
    "extract_sentences",
    "kb_prompt",
    "num_tokens_from_string",
    "ThinkingEngine",
    "AnswerValidator",
    "DualPathSearcher",
    "QueryGenerator",
    "CommunityAwareSearchEnhancer",
    "DynamicKnowledgeGraphBuilder",
    "EvidenceChainTracker",
]