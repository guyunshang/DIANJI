from typing import Optional  # 或者导入 Union
from langchain_core.language_models.chat_models import BaseChatModel

import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

"""
引用格式化模块
"""
from typing import Iterable, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from graphrag_agent.config.prompts import CITATION_FORMAT_PROMPT
from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.agents.multi_agent.core.retrieval_result import RetrievalResult


class CitationFormatter:
    """
    引用格式化器，将RetrievalResult转换成符合要求的引用列表
    """

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self._llm = llm or get_llm_model()

    def format_references(
        self,
        retrieval_results: Iterable[RetrievalResult],
        citation_style: str = "default",
    ) -> str:
        serialized = [
            result.to_dict()
            for result in retrieval_results
        ]
        prompt = CITATION_FORMAT_PROMPT.format(
            retrieval_results=serialized,
            citation_style=citation_style,
        )
        message: BaseMessage = self._llm.invoke(prompt)  # type: ignore[assignment]
        content = getattr(message, "content", None) or str(message)
        return content.strip()
