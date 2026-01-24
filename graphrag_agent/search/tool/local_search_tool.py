import sys
from pathlib import Path

# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from typing import List, Dict, Any
import re
import time
import ast
from langchain.docstore.document import Document
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graphrag_agent.config.prompts import (
    LC_SYSTEM_PROMPT,
    LOCAL_SEARCH_CONTEXT_PROMPT,
    ENTITY_EXTRACTION_WITH_SCHEMA_PROMPT,
    SEARCH_RESULT_JUDGE_PROMPT
)
from graphrag_agent.config.settings import (
    lc_description,
    entity_definitions,
    relationship_definitions,
    response_type
)
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.search.local_search import LocalSearch
from graphrag_agent.config.settings import lc_description, response_type


class LocalSearchTool(BaseSearchTool):
    """本地搜索工具，基于向量检索实现社区内部的精确查询"""

    def __init__(self):
        # 必须先于其他初始化调用 super().__init__
        super().__init__(cache_dir="./cache/local_search")

        self.chat_history = []
        self.local_searcher = LocalSearch(self.llm, self.embeddings)
        self.retriever = self.local_searcher.as_retriever()

        # 初始化处理链
        self._setup_chains()

    # --- 实现基类要求的抽象方法 ---
    def _setup_chains(self):
        """初始化生成回答的 Chain"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", LOCAL_SEARCH_CONTEXT_PROMPT),
        ])
        self.question_answer_chain = prompt | self.llm | StrOutputParser()

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取关键词接口"""
        return {"low_level": [query], "high_level": [query]}

    def search(self, query_input: Any) -> str:
        """标准搜索接口，返回字符串答案"""
        res = self.structured_search(query_input)
        return res.get("answer", "未找到相关信息。")


    # --- A/B 路径检索 ---
    def _parse_custom_extraction(self, text: Any) -> Dict[str, Any]:
        import re
        if isinstance(text, list):
            text = text[0] if text else ""

        text = text.replace('\n', ' ').replace('\r', ' ')

        def get_clean_list(label):
            pattern = rf"{label}[:：\s]*\[(.*?)\]"
            match = re.search(pattern, text)
            if match:
                content = match.group(1)
                items = re.split(r'[,，、\s]+', content)
                return [i.strip() for i in items if i.strip()]
            return []

        # 1. 提取核心总结词
        core_match = re.search(r"核心总结词[:：\s]*\[(.*?)\]", text)
        core_summary = core_match.group(1).strip() if core_match else ""

        if not core_summary and "核心总结词" in text:
            try:
                core_summary = text.split("核心总结词")[1].split("[")[1].split("]")[0].strip()
            except:
                pass

        # 2. 解析所有问题信息三元组 [A-B-C]
        triples = []
        triple_section = re.search(r"问题信息三元组[:：\s]*(.*)", text)
        if triple_section:
            triple_text = triple_section.group(1)
            items = re.findall(r"\[(.*?)-(.*?)-(.*?)\]", triple_text)
            for sub, pred, obj in items:
                triples.append({
                    "subject": sub.strip(),
                    "predicate": pred.strip(),
                    "object": obj.strip()
                })

        return {
            "entity_names": get_clean_list("实体名称"),
            "entity_types": get_clean_list("实体类型"),
            "relation_types": get_clean_list("关系类型"),
            "core_summary": core_summary,
            "triples": triples  # 确保包含此字段供 A 路使用
        }

    def structured_search(self, query_input: Any) -> Dict[str, Any]:
        overall_start = time.time()
        parsed_input = self._normalize_input(query_input)
        query = parsed_input["query"]

        # 1. LLM 提取 (完全保留原有逻辑与 Prompt 填充)
        raw_res = self.llm.invoke(ENTITY_EXTRACTION_WITH_SCHEMA_PROMPT.format(
            query=query,
            entity_definitions="\n".join([f"- {k}: {v}" for k, v in entity_definitions.items()]),
            relationship_types="\n".join([f"- {k}: {v}" for k, v in relationship_definitions.items()])
        )).content

        extracted = self._parse_custom_extraction(raw_res)

        core_summary = extracted["core_summary"]
        if not core_summary or len(core_summary) < 2:
            core_summary = query

        print(f"\n[DEBUG] 向量搜索使用的核心词: '{core_summary}'")

        # 2. 执行检索
        # A路：执行你修改后的三元组路径检索
        res_a = self.local_searcher.keyword_graph_search(
            extracted["entity_names"],
            extracted.get("triples", [])
        )
        # B路：切换为直连社区与文本块的检索函数
        res_b = self.local_searcher.vector_search_communities_and_chunks(core_summary)

        # === 3. 数据解析与 ID/文件名 提取 (保留原有变量命名，解决哈希编码问题) ===
        # A 路提取：优先取 fileName 属性
        ids_a = [c.get('fileName') or c['id'] for c in res_a.get('Chunks', [])]
        entities_a = res_a.get('HitEntities', [])

        ids_b = []
        entities_b = []  # 用于保留 B 路命中的实体/社区标识
        text_b_refined = ""

        # 解析 B 路结果 (res_b 现在包含 Communities 和 Chunks)
        if res_b:
            # 处理文本块
            for c in res_b.get('Chunks', []):
                ids_b.append(c.get('fileName') or c['id'])
                text_b_refined += c['text'] + "\n"

            # 处理社区摘要
            for com in res_b.get('Communities', []):
                # 将社区 ID/名称 放入 entities_b 以兼容原有的“命中实体”打印
                entities_b.append(com.get('id'))
                text_b_refined += com['text'] + "\n"

        # === 4. 打印调试信息 (严格保留所有原有打印格式与逻辑) ===
        unique_ids_b = list(set(ids_b))
        all_hit_ids = list(set(ids_a + unique_ids_b))
        all_hit_entities = list(set(entities_a + entities_b))

        hit_groups_a = res_a.get('HitGroups', {})

        print("\n" + "=" * 50)

        # 4.1 打印 A 路三元组分组结果
        for label, entities in hit_groups_a.items():
            if label != "其他实体" and entities:
                unique_entities = list(set(entities))
                print(f"【A路】{label}：{unique_entities}")

        # 4.2 打印 A 路其他实体
        others = hit_groups_a.get("其他实体", [])
        if others:
            print(f"【A路】其他实体：{list(set(others))}")

        # 4.3 打印 B 路命中情况 (entities_b 此时包含社区标识)
        if entities_b:
            print(f"【B路(向量)命中实体】: {list(set(entities_b))}")

        # 4.4 打印汇总统计
        print(f"【当前汇总命中实体名称】: {all_hit_entities}")
        print(f"【汇总去重后 Chunk ID 数量】: {len(all_hit_ids)}")
        print(f"【所有 Chunk ID 列表】: {all_hit_ids}")  # 此时 ID 列表已优化为文件名
        print("=" * 50 + "\n")

        # 5. 合并上下文与生成回答 (保留原有 Judge 裁决逻辑)
        text_a = "\n".join([c['text'] for c in res_a.get('Chunks', [])])
        text_b = text_b_refined

        final_context = ""
        if text_a and text_b and text_a.strip() != text_b.strip():
            # 使用 LLM 裁决最相关的路径
            judge_res = self.llm.invoke(SEARCH_RESULT_JUDGE_PROMPT.format(
                query=query, result_a=text_a[:1500], result_b=text_b[:1500]
            )).content
            final_context = text_a if "A" in judge_res else text_b
        else:
            final_context = text_a or text_b

        if not final_context.strip():
            return {"answer": "抱歉，在知识库中未检索到相关详细信息。", "query": query}

        # 6. 调用生成链 (保持原代码处理)
        answer = self.question_answer_chain.invoke({
            "context": final_context,
            "input": query,
            "response_type": response_type
        })

        return {
            "answer": answer,
            "query": query,
            "retrieval_results": all_hit_ids  # 返回包含文件名的列表
        }

    def _normalize_input(self, query_input: Any) -> Dict[str, Any]:
        if isinstance(query_input, str): return {"query": query_input}
        if isinstance(query_input, dict): return query_input
        return {"query": str(query_input)}


    def get_structured_tool(self) -> BaseTool:
        """返回结构化工具"""
        outer = self

        class LocalSearchStructuredTool(BaseTool):
            name: str = "local_search_structured"
            description: str = "结构化本地搜索工具，用于电力知识精准查询。"

            def _run(self_tool, query: Any, **kwargs: Any) -> Dict[str, Any]:
                payload = query if isinstance(query, dict) else {"query": query}
                payload.update(kwargs)
                return outer.structured_search(payload)

        return LocalSearchStructuredTool()

    def close(self):
        pass