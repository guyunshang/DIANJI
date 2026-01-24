import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

# 搜索模块初始化文件
# 包含基础搜索类和高级搜索工具类

# 导出主要类
from graphrag_agent.search.local_search import LocalSearch
from graphrag_agent.search.global_search import GlobalSearch

# 导出工具类
from graphrag_agent.search.tool.local_search_tool import LocalSearchTool
from graphrag_agent.search.tool.global_search_tool import GlobalSearchTool
from graphrag_agent.search.tool.hybrid_tool import HybridSearchTool
from graphrag_agent.search.tool.naive_search_tool import NaiveSearchTool
from graphrag_agent.search.tool.deep_research_tool import DeepResearchTool

__all__ = [
    "LocalSearch",
    "GlobalSearch",
    "LocalSearchTool",
    "GlobalSearchTool",
    "HybridSearchTool",
    "NaiveSearchTool",
    "DeepResearchTool"
]