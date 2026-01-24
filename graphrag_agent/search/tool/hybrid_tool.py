import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

import time
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
from neo4j import Result

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graphrag_agent.config.prompts import (
    LC_SYSTEM_PROMPT,
    HYBRID_TOOL_QUERY_PROMPT,
    LOCAL_SEARCH_KEYWORD_PROMPT,
)
from graphrag_agent.config.settings import gl_description, response_type, HYBRID_SEARCH_SETTINGS
from graphrag_agent.search.tool.reasoning.nlp import NLPProcessor
from graphrag_agent.search.tool.base import BaseSearchTool
from graphrag_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from graphrag_agent.search.retrieval_adapter import (
    create_retrieval_metadata,
    create_retrieval_result,
    merge_retrieval_results,
    results_from_entities,
    results_from_relationships,
    results_to_payload,
)


class HybridSearchTool(BaseSearchTool):
    """
    混合搜索工具，实现类似LightRAG的双级检索策略
    结合了局部细节检索和全局主题检索
    """
    
    def __init__(self):
        """初始化混合搜索工具"""
        # 检索参数
        self.entity_limit = HYBRID_SEARCH_SETTINGS["entity_limit"]
        self.max_hop_distance = HYBRID_SEARCH_SETTINGS["max_hop_distance"]
        self.top_communities = HYBRID_SEARCH_SETTINGS["top_communities"]
        self.batch_size = HYBRID_SEARCH_SETTINGS["batch_size"]
        self.community_level = HYBRID_SEARCH_SETTINGS["community_level"]
        
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/hybrid_search")

        # 设置处理链
        self._setup_chains()
    
    def _setup_chains(self):
        """设置处理链"""
        # 创建主查询处理链 - 用于生成最终答案
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", HYBRID_TOOL_QUERY_PROMPT),
        ])
        
        # 链接到LLM
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()
        
        # 关键词提取链
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", LOCAL_SEARCH_KEYWORD_PROMPT),
            ("human", "{query}"),
        ])
        
        self.keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()


    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        【本体感知重写版】利用专业电力本体提取关键词，并自动映射到双级通道。
        """
        # 1. 检查缓存
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords

        try:
            llm_start = time.time()

            # 2. 调用我们之前定义的本体感知处理器
            nlp = NLPProcessor(llm=self.llm)
            professional_keywords = nlp.extract_keywords(query)

            print(f"\n" + "=" * 30)
            print(f"【搜索优化】原始问题: {query}")
            print(f"关键词: {professional_keywords}")
            print("=" * 30 + "\n")

            # 3. 统计性能
            self.performance_metrics["llm_time"] += time.time() - llm_start

            # 4. 统一分发：将专业词汇同时喂给局部和全局检索
            keywords = {
                "low_level": professional_keywords,
                "high_level": professional_keywords
            }

            # 5. 缓存并返回
            self.cache_manager.set(f"keywords:{query}", keywords)
            return keywords

        except Exception as e:
            print(f"本体感知提取失败: {e}，回退至基础查询")
            return {"low_level": [query], "high_level": [query]}
    
    def db_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        执行Cypher查询并返回结果
        
        参数:
            cypher: Cypher查询语句
            params: 查询参数
            
        返回:
            pandas.DataFrame: 查询结果
        """
        return self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_=Result.to_df
        )
    
    def _vector_search(self, query: str, limit: int = 5) -> List[str]:
        """
        使用基类的向量搜索方法
        
        参数:
            query: 查询字符串
            limit: 最大结果数
            
        返回:
            List[str]: 实体ID列表
        """
        return self.vector_search(query, limit)

    def _fallback_text_search(self, query: str, limit: int = 5) -> List[str]:
        """
        基于文本匹配的备用搜索方法
        
        参数:
            query: 搜索查询
            limit: 最大返回结果数
            
        返回:
            List[str]: 匹配实体ID列表
        """
        try:
            # 构建全文搜索查询
            cypher = """
            MATCH (e:__Entity__)
            WHERE e.id CONTAINS $query OR e.description CONTAINS $query
            RETURN e.id AS id
            LIMIT $limit
            """
            
            results = self.db_query(cypher, {
                "query": query,
                "limit": limit
            })
            
            if not results.empty:
                return results['id'].tolist()
            else:
                return []
                
        except Exception as e:
            print(f"文本搜索也失败: {e}")
            return []
    
    def _retrieve_low_level_content(self, query: str, keywords: List[str]) -> Tuple[str, List[RetrievalResult]]:
        """
        检索低级内容（具体实体和关系）
        
        参数:
            query: 查询字符串
            keywords: 低级关键词列表
            
        返回:
            Tuple[str, List[RetrievalResult]]: 格式化内容及对应证据
        """
        query_start = time.time()
        retrieval_results: List[RetrievalResult] = []
        
        # 首先使用关键词查询获取相关实体
        entity_ids = []
        
        if keywords:
            keyword_params = {}
            keyword_conditions = []
            
            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                keyword_params[param_name] = keyword
                keyword_conditions.append(f"e.id CONTAINS ${param_name} OR e.description CONTAINS ${param_name}")
            
            # 构建查询
            if keyword_conditions:
                keyword_query = """
                MATCH (e:__Entity__)
                WHERE """ + " OR ".join(keyword_conditions) + """
                RETURN e.id AS id
                LIMIT $limit
                """
                
                try:
                    keyword_results = self.db_query(keyword_query, 
                                                {**keyword_params, "limit": self.entity_limit})
                    if not keyword_results.empty:
                        entity_ids = keyword_results['id'].tolist()
                except Exception as e:
                    print(f"关键词查询失败: {e}")
        
        # 如果关键词搜索没有结果或没有提供关键词，尝试使用向量搜索
        if not entity_ids:
            try:
                # 使用我们的自定义向量搜索方法
                vector_entity_ids = self._vector_search(query, limit=self.entity_limit)
                if vector_entity_ids:
                    entity_ids = vector_entity_ids
            except Exception as e:
                print(f"向量搜索失败: {e}")
        
        # 如果仍然没有实体，使用基本文本匹配
        if not entity_ids:
            try:
                entity_ids = self._fallback_text_search(query, limit=self.entity_limit)
            except Exception as e:
                print(f"文本搜索失败: {e}")
        
        # 如果仍然没有实体，返回空内容
        if not entity_ids:
            self.performance_metrics["query_time"] += time.time() - query_start
            return "没有找到相关的低级内容。", retrieval_results
        
        # 获取实体信息 - 不使用多跳关系以避免复杂查询
        entity_query = """
        // 从种子实体开始
        MATCH (e:__Entity__)
        WHERE e.id IN $entity_ids
        
        RETURN collect({
            id: e.id, 
            type: CASE WHEN size(labels(e)) > 1 
                     THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0] 
                     ELSE 'Unknown' 
                  END, 
            description: e.description
        }) AS entities
        """
        
        # 获取关系信息 - 分别查询，避免复杂路径
        relation_query = """
        // 查找实体间的关系
        MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
        WHERE e1.id IN $entity_ids 
          AND e2.id IN $entity_ids
          AND e1.id < e2.id  // 避免重复关系
        
        RETURN collect({
            start: e1.id, 
            type: type(r), 
            end: e2.id,
            description: CASE WHEN r.description IS NULL THEN '' ELSE r.description END
        }) AS relationships
        """
        
        # 获取文本块信息
        chunk_query = """
        // 查找包含这些实体的文本块
        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
        WHERE e.id IN $entity_ids
        
        RETURN collect(DISTINCT {
            id: c.id, 
            text: c.text
        })[0..5] AS chunks
        """
        
        try:
            # 获取实体信息
            entity_results = self.db_query(entity_query, {"entity_ids": entity_ids})
            
            # 获取关系信息
            relation_results = self.db_query(relation_query, {"entity_ids": entity_ids})
            
            # 获取文本块信息
            chunk_results = self.db_query(chunk_query, {"entity_ids": entity_ids})
            
            self.performance_metrics["query_time"] += time.time() - query_start
            
            # 构建结果
            low_level = []
            
            # 添加实体信息
            if not entity_results.empty and 'entities' in entity_results.columns:
                entities = entity_results.iloc[0]['entities']
                if entities:
                    low_level.append("### 相关实体")
                    entity_dicts: List[Dict[str, Any]] = []
                    for entity in entities:
                        entity_desc = f"- **{entity['id']}** ({entity['type']}): {entity['description']}"
                        low_level.append(entity_desc)
                        entity_dicts.append(
                            {
                                "id": entity["id"],
                                "description": entity["description"],
                                "confidence": 0.65,
                                "type": entity["type"],
                            }
                        )
                    retrieval_results.extend(
                        results_from_entities(
                            entity_dicts,
                            source="hybrid_search",
                            confidence=0.65,
                        )
                    )
            
            # 添加关系信息
            if not relation_results.empty and 'relationships' in relation_results.columns:
                relationships = relation_results.iloc[0]['relationships']
                if relationships:
                    low_level.append("\n### 实体关系")
                    relationship_dicts: List[Dict[str, Any]] = []
                    for rel in relationships:
                        rel_desc = f"- **{rel['start']}** -{rel['type']}-> **{rel['end']}**: {rel['description']}"
                        low_level.append(rel_desc)
                        relationship_dicts.append(
                            {
                                "start": rel["start"],
                                "end": rel["end"],
                                "type": rel["type"],
                                "description": rel.get("description", ""),
                                "confidence": 0.6,
                                "weight": 0.6,
                            }
                        )
                    retrieval_results.extend(
                        results_from_relationships(
                            relationship_dicts,
                            source="hybrid_search",
                            confidence=0.6,
                        )
                    )
            
            # 添加文本块信息
            if not chunk_results.empty and 'chunks' in chunk_results.columns:
                chunks = chunk_results.iloc[0]['chunks']
                if chunks:
                    low_level.append("\n### 相关文本")
                    for chunk in chunks:
                        chunk_text = f"- ID: {chunk['id']}\n  内容: {chunk['text']}"
                        low_level.append(chunk_text)
                        retrieval_results.append(
                            create_retrieval_result(
                                evidence=chunk.get("text", ""),
                                source="hybrid_search",
                                granularity="Chunk",
                                metadata=create_retrieval_metadata(
                                    source_id=str(chunk.get("id")),
                                    source_type="chunk",
                                    confidence=0.7,
                                    extra={"raw_chunk": chunk},
                                ),
                                score=0.7,
                            )
                        )
            
            if not low_level:
                return "没有找到相关的低级内容。", retrieval_results
                
            return "\n".join(low_level), retrieval_results
        except Exception as e:
            self.performance_metrics["query_time"] += time.time() - query_start
            print(f"实体查询失败: {e}")
            return "查询实体信息时出错。", retrieval_results
    
    def _retrieve_high_level_content(self, query: str, keywords: List[str]) -> Tuple[str, List[RetrievalResult]]:
        """
        检索高级内容（社区和主题概念）
        
        参数:
            query: 查询字符串
            keywords: 高级关键词列表
            
        返回:
            Tuple[str, List[RetrievalResult]]: 格式化内容及对应证据
        """
        query_start = time.time()
        retrieval_results: List[RetrievalResult] = []
        
        # 构建关键词条件
        keyword_conditions = []
        params = {"level": self.community_level, "limit": self.top_communities}
        
        if keywords:
            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                params[param_name] = keyword
                keyword_conditions.append(f"c.summary CONTAINS ${param_name} OR c.full_content CONTAINS ${param_name}")
        
        # 构建查询
        community_query = """
        // 使用关键词过滤社区
        MATCH (c:__Community__ {level: $level})
        """
        
        if keyword_conditions:
            community_query += "WHERE " + " OR ".join(keyword_conditions)
        else:
            # 如果没有关键词，则使用查询文本
            params["query"] = query
            community_query += "WHERE c.summary CONTAINS $query OR c.full_content CONTAINS $query"
        
        # 添加排序和限制
        community_query += """
        WITH c
        ORDER BY CASE WHEN c.community_rank IS NULL THEN 0 ELSE c.community_rank END DESC
        LIMIT $limit
        RETURN c.id AS id, c.summary AS summary
        """
        
        try:
            community_results = self.db_query(community_query, params)
            
            self.performance_metrics["query_time"] += time.time() - query_start
            
            # 处理结果
            if community_results.empty:
                return "没有找到相关的高级内容。", retrieval_results
                
            # 构建格式化的高级内容
            high_level = ["### 相关主题概念"]
            
            for _, row in community_results.iterrows():
                community_desc = f"- **社区 {row['id']}**:\n  {row['summary']}"
                high_level.append(community_desc)
                retrieval_results.append(
                    create_retrieval_result(
                        evidence=row.get("summary", ""),
                        source="hybrid_search",
                        granularity="DO",
                        metadata=create_retrieval_metadata(
                            source_id=str(row.get("id")),
                            source_type="community",
                            confidence=0.6,
                            community_id=str(row.get("id")),
                            extra={"raw_community": row.to_dict()},
                        ),
                        score=0.6,
                    )
                )
            
            return "\n".join(high_level), retrieval_results
        except Exception as e:
            self.performance_metrics["query_time"] += time.time() - query_start
            print(f"社区查询失败: {e}")
            return "查询社区信息时出错。", retrieval_results
    
    def structured_search(self, query_input: Any) -> Dict[str, Any]:
        """
        执行混合搜索，返回包含证据与答案的结构化结果。
        """
        overall_start = time.time()
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            # 支持直接传入分类的关键词
            low_keywords = query_input.get("low_level_keywords", [])
            high_keywords = query_input.get("high_level_keywords", [])
        else:
            query = str(query_input)
            # 提取关键词
            keywords = self.extract_keywords(query)
            low_keywords = keywords.get("low_level", [])
            high_keywords = keywords.get("high_level", [])
        
        # 检查缓存
        cache_key = query
        if low_keywords or high_keywords:
            cache_key = self.cache_manager.key_strategy.generate_key(
                query, 
                low_level_keywords=low_keywords, 
                high_level_keywords=high_keywords
            )
            
        cached_result = self.cache_manager.get(cache_key)
        if isinstance(cached_result, dict):
            return cached_result
        
        try:
            # 1. 检索低级内容（实体和关系）
            low_level_content, low_evidence = self._retrieve_low_level_content(query, low_keywords)
            
            # 2. 检索高级内容（社区和主题）
            high_level_content, high_evidence = self._retrieve_high_level_content(query, high_keywords)
            
            # 3. 生成最终答案
            llm_start = time.time()
            
            # 调用LLM生成最终答案
            answer = self.query_chain.invoke({
                "query": query,
                "low_level": low_level_content,
                "high_level": high_level_content,
                "response_type": response_type
            })
            
            self.performance_metrics["llm_time"] += time.time() - llm_start
            
            all_evidence = merge_retrieval_results(low_evidence, high_evidence)
            structured_result = {
                "query": query,
                "low_level_content": low_level_content,
                "high_level_content": high_level_content,
                "final_answer": answer if answer else "未找到相关信息",
                "retrieval_results": results_to_payload(all_evidence),
            }
            
            # 缓存结果
            self.cache_manager.set(
                cache_key, 
                structured_result, 
                low_level_keywords=low_keywords,
                high_level_keywords=high_keywords
            )
            
            self.performance_metrics["total_time"] = time.time() - overall_start

            return structured_result
            
        except Exception as e:
            error_msg = f"搜索过程中出现错误: {str(e)}"
            print(error_msg)
            return {
                "query": query,
                "low_level_content": "",
                "high_level_content": "",
                "final_answer": error_msg,
                "retrieval_results": [],
                "error": error_msg,
            }
    
    def search(self, query_input: Any) -> str:
        """
        执行混合搜索，结合低级和高级内容
        
        参数:
            query_input: 字符串查询或包含查询和关键词的字典
            
        返回:
            str: 生成的最终答案
        """
        structured = self.structured_search(query_input)
        return structured.get("final_answer", "未找到相关信息")

    def get_global_tool(self) -> BaseTool:
        """
        获取全局搜索工具，采用属性注入方式解决作用域问题
        """
        # 获取当前的 HybridSearchTool 实例
        current_hybrid_instance = self

        class GlobalSearchTool(BaseTool):
            # 定义工具的基本属性
            name: str = "global_retriever"
            description: str = gl_description

            # 【核心修复】显式定义一个字段来持有外部 HybridSearchTool 的引用
            # 使用 Any 避免复杂的循环导入检查
            hybrid_instance: Any

            def _run(self, query: Any) -> str:  # 将 str 改为 Any 以防万一
                """
                执行混合搜索
                """
                # 如果模型传过来的是 {'title': '...', 'type': 'string'} 这种字典
                if isinstance(query, dict):
                    if 'title' in query:
                        query = query['title']
                    elif 'query' in query:
                        query = query['query']
                    else:
                        # 兜底：如果是个字典但没找到 key，转成 json 字符串
                        query = json.dumps(query, ensure_ascii=False)

                # 确保 query 最终是字符串
                query = str(query)
                nlp = NLPProcessor(llm=self.hybrid_instance.llm)
                professional_keywords = nlp.extract_keywords(query)

                # 3. 打印调试信息
                print(f"\n" + "=" * 30)
                print(f"【搜索优化】原始问题: {query}")
                print(f"关键词: {professional_keywords}")
                print("=" * 30 + "\n")

                # 4. 构造优化参数
                optimized_params = {
                    "query": query,
                    "high_level_keywords": professional_keywords,
                    "low_level_keywords": professional_keywords
                }

                try:
                    # 5. 调用外部实例的 search 方法
                    return self.hybrid_instance.search(optimized_params)
                except Exception as e:
                    print(f"搜索优化执行异常: {e}，正在尝试原始搜索...")
                    return self.hybrid_instance.search(query)

            def _arun(self, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")

        # 【关键修复】在实例化工具时，将 self 传给 hybrid_instance 字段
        return GlobalSearchTool(hybrid_instance=current_hybrid_instance)
    
    def close(self):
        """关闭资源"""
        super().close()
