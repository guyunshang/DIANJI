import sys
from pathlib import Path

# 在文件开头添加项目根目录到系统路径
# 修改为 .parent.parent.parent 以准确指向项目主根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

"""
搜索工具注册表

集中列出 LangChain 可调用的搜索类，目前仅保留与 Hybrid Agent 相关的核心检索工具。
"""

from typing import Any, Dict, Type

from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.tool.local_search_tool import LocalSearchTool
from graphrag_agent.search.tool.global_search_tool import GlobalSearchTool
from graphrag_agent.search.tool.hybrid_tool import HybridSearchTool

# 核心工具注册表：仅保留 Hybrid Agent 依赖的本地、全局及混合搜索工具
TOOL_REGISTRY: Dict[str, Type[BaseSearchTool]] = {
    "local_search": LocalSearchTool,
    "global_search": GlobalSearchTool,
    "hybrid_search": HybridSearchTool,
}

# 额外工具注册表：已移除链式探索、假设生成等深度研究专用工具，目前设为空
EXTRA_TOOL_FACTORIES: Dict[str, Any] = {}


def get_tool_class(tool_name: str) -> Type[BaseSearchTool]:
    """根据名称获取工具类，若不存在则抛出 KeyError"""
    return TOOL_REGISTRY[tool_name]


def available_tools() -> Dict[str, Type[BaseSearchTool]]:
    """返回注册表的浅拷贝"""
    return dict(TOOL_REGISTRY)


def available_extra_tools() -> Dict[str, Any]:
    """返回额外工具工厂的浅拷贝"""
    return dict(EXTRA_TOOL_FACTORIES)


def create_extra_tool(tool_name: str) -> Any:
    """根据名称创建额外工具实例"""
    if tool_name not in EXTRA_TOOL_FACTORIES:
        raise KeyError(f"工具 {tool_name} 不在额外工具注册表中")
    factory = EXTRA_TOOL_FACTORIES[tool_name]
    return factory()


__all__ = [
    "TOOL_REGISTRY",
    "EXTRA_TOOL_FACTORIES",
    "get_tool_class",
    "available_tools",
    "available_extra_tools",
    "create_extra_tool",
]