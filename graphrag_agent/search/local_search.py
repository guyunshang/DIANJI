import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from typing import Dict, Any
import pandas as pd
from neo4j import Result
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graphrag_agent.config.prompts import LC_SYSTEM_PROMPT, LOCAL_SEARCH_CONTEXT_PROMPT
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import LOCAL_SEARCH_SETTINGS

class LocalSearch:
    """
    本地搜索类：使用Neo4j和LangChain实现基于向量检索的本地搜索功能
    
    该类通过向量相似度搜索在知识图谱中查找相关内容，并生成回答
    主要功能包括：
    1. 基于向量相似度的文本检索
    2. 社区内容和关系的检索
    3. 使用LLM生成最终答案
    """
    
    def __init__(self, llm, embeddings, response_type: str = "多个段落"):
        """
        初始化本地搜索类
        
        参数:
            llm: 大语言模型实例
            embeddings: 向量嵌入模型
            response_type: 响应类型，默认为"多个段落"
        """
        # 保存模型实例和配置
        self.llm = llm
        self.embeddings = embeddings
        self.response_type = response_type
        
        # 获取数据库连接管理器
        db_manager = get_db_manager()
        self.driver = db_manager.get_driver()
        
        # 设置检索参数
        self.top_chunks = LOCAL_SEARCH_SETTINGS["top_chunks"]
        self.top_communities = LOCAL_SEARCH_SETTINGS["top_communities"]
        self.top_outside_rels = LOCAL_SEARCH_SETTINGS[
            "top_outside_relationships"
        ]
        self.top_inside_rels = LOCAL_SEARCH_SETTINGS[
            "top_inside_relationships"
        ]
        self.top_entities = LOCAL_SEARCH_SETTINGS["top_entities"]
        self.index_name = LOCAL_SEARCH_SETTINGS["index_name"]
        
        # 初始化社区节点权重
        self._init_community_weights()
        
        # 配置Neo4j URI和认证信息
        self.neo4j_uri = db_manager.neo4j_uri
        self.neo4j_username = db_manager.neo4j_username
        self.neo4j_password = db_manager.neo4j_password
        
    def _init_community_weights(self):
        """初始化Neo4j中社区节点的权重"""
        self.db_query("""
        MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:MENTIONS]-(c)
        WITH n, count(distinct c) AS chunkCount
        SET n.weight = chunkCount
        """)
        
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
        
    @property
    def retrieval_query(self) -> str:
        """向量搜索(B路)的 Cypher 模板：修复作用域报错"""
        return """
        WITH collect(node) as nodes
        WITH nodes, 
        collect {
            UNWIND nodes as n
            MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
            WITH distinct c, count(distinct n) as freq
            RETURN {id:c.id, text: c.text, fileName: c.fileName} AS chunkText // 增加 fileName
            ORDER BY freq DESC
            LIMIT $topChunks
        } AS text_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
            WITH distinct c, c.community_rank as rank, c.weight AS weight
            RETURN {id: c.id, summary: c.summary, title: c.title} 
            ORDER BY rank, weight DESC
            LIMIT $topCommunities
        } AS report_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE NOT m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topOutsideRels
        } as outsideRels,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topInsideRels
        } as insideRels,
        collect {
            UNWIND nodes as n
            RETURN n.description AS descriptionText
        } as entities
        RETURN {
            Chunks: text_mapping, 
            Reports: report_mapping, 
            Relationships: outsideRels + insideRels, 
            Entities: entities,
            HitEntities: [n IN nodes | n.id] 
        } AS text, 1.0 AS score, {} AS metadata
        """
    
    def as_retriever(self, **kwargs):
        """
        返回检索器实例，用于链式调用
        
        返回:
            检索器实例
        """
        # 生成包含所有检索参数的查询
        final_query = self.retrieval_query.replace("$topChunks", str(self.top_chunks))\
            .replace("$topCommunities", str(self.top_communities))\
            .replace("$topOutsideRels", str(self.top_outside_rels))\
            .replace("$topInsideRels", str(self.top_inside_rels))

        db_manager = get_db_manager()
        
        # 初始化向量存储
        vector_store = Neo4jVector.from_existing_index(
            self.embeddings,
            url=db_manager.neo4j_uri,
            username=db_manager.neo4j_username,
            password=db_manager.neo4j_password,
            index_name=self.index_name,
            text_node_property="description",
            retrieval_query=final_query
        )
        
        # 返回检索器
        return vector_store.as_retriever(
            search_kwargs={"k": self.top_entities}
        )
        
    def search(self, query: str) -> str:
        """
        执行本地搜索
        
        参数:
            query: 搜索查询字符串
            
        返回:
            str: 生成的最终答案
        """
        # 初始化对话提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", LOCAL_SEARCH_CONTEXT_PROMPT),
        ])
        
        # 创建搜索链
        chain = prompt | self.llm | StrOutputParser()
        
        # 初始化向量存储
        vector_store = Neo4jVector.from_existing_index(
            self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=self.index_name,
            text_node_property="description",
            retrieval_query=self.retrieval_query
        )
        
        # 执行相似度搜索
        docs = vector_store.similarity_search(
            query,
            k=self.top_entities,
            params={
                "topChunks": self.top_chunks,
                "topCommunities": self.top_communities,
                "topOutsideRels": self.top_outside_rels,
                "topInsideRels": self.top_inside_rels,
            }
        )
        
        # 使用LLM生成响应
        response = chain.invoke({
            "context": docs[0].page_content if docs else "",
            "input": query,
            "response_type": self.response_type
        })
        
        return response
        
    def close(self):
        """关闭Neo4j驱动连接"""
        pass
        
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def get_aligned_node_id(self, name: str) -> str:
        """
        三级节点对齐：完全匹配(最高优先) -> 正则模糊 -> 向量相似度
        """
        if not name or name in ["?", "？"]:
            return ""

        # 1. 【第一准则】：完全匹配查询 (完全对应)
        res = self.db_query("MATCH (e:__Entity__ {id: $name}) RETURN e.id AS id", {"name": name})
        if not res.empty:
            return str(res.iloc[0]['id'])

        # 2. 第二准则：正则模糊匹配
        regex_query = "MATCH (e:__Entity__) WHERE e.id =~ ('(?i).*' + $name + '.*') RETURN e.id AS id LIMIT 1"
        res = self.db_query(regex_query, {"name": name})
        if not res.empty:
            return str(res.iloc[0]['id'])

        # 3. 第三准则：向量相似度匹配
        try:
            query_embedding = self.embeddings.embed_query(name)
            vector_cypher = """
            CALL db.index.vector.queryNodes($index, 1, $embedding) 
            YIELD node, score 
            RETURN node.id AS id
            """
            res = self.db_query(vector_cypher, {
                "index": self.index_name,
                "embedding": query_embedding
            })
            if not res.empty:
                return str(res.iloc[0]['id'])
        except Exception as e:
            print(f"[Warning] 向量对齐异常: {e}")
        return ""

    def keyword_graph_search(self, entity_names: list, triples: list) -> dict:
        """
        A 路检索：基于三元组进行路径探索，并对命中实体进行分组
        """
        all_hit_entity_ids = set()
        # 初始化分组字典，默认包含“其他实体”
        hit_groups = {"其他实体": []}

        # 1. 第一步：对齐提取的原始实体名，归类为“其他实体”
        for name in entity_names:
            aligned_id = self.get_aligned_node_id(name)
            if aligned_id:
                all_hit_entity_ids.add(aligned_id)
                hit_groups["其他实体"].append(aligned_id)

        # 2. 第二步：遍历所有三元组进行路径补全
        for triple in triples:
            sub, pred, obj = triple['subject'], triple['predicate'], triple['object']

            # 情况 A: [已知端 - 关系 - ?] -> 寻找所有下游
            if sub not in ["?", "？"] and obj in ["?", "？"]:
                label = f"{sub}-{pred}"  # 构造分组标签
                if label not in hit_groups: hit_groups[label] = []

                start_node = self.get_aligned_node_id(sub)
                if start_node:
                    all_hit_entity_ids.add(start_node)
                    # 确保起点实体也出现在“其他实体”中
                    if start_node not in hit_groups["其他实体"]:
                        hit_groups["其他实体"].append(start_node)

                    # 提取所有匹配节点，不设 LIMIT
                    path_query = "MATCH (n:__Entity__ {id: $id})-[r]->(m:__Entity__) WHERE type(r) CONTAINS $p RETURN m.id AS id"
                    res = self.db_query(path_query, {"id": start_node, "p": pred})
                    found_ids = res['id'].astype(str).tolist()
                    all_hit_entity_ids.update(found_ids)
                    hit_groups[label].extend(found_ids)

            # 情况 B: [? - 关系 - 已知端] -> 寻找所有上游
            elif obj not in ["?", "？"] and sub in ["?", "？"]:
                label = f"{pred}-{obj}"  # 构造分组标签
                if label not in hit_groups: hit_groups[label] = []

                end_node = self.get_aligned_node_id(obj)
                if end_node:
                    all_hit_entity_ids.add(end_node)
                    if end_node not in hit_groups["其他实体"]:
                        hit_groups["其他实体"].append(end_node)

                    path_query = "MATCH (m:__Entity__)-[r]->(n:__Entity__ {id: $id}) WHERE type(r) CONTAINS $p RETURN m.id AS id"
                    res = self.db_query(path_query, {"id": end_node, "p": pred})
                    found_ids = res['id'].astype(str).tolist()
                    all_hit_entity_ids.update(found_ids)
                    hit_groups[label].extend(found_ids)

        # 3. 第三步：基于全量实体集计算 Chunk 覆盖度
        if not all_hit_entity_ids:
            return {"Chunks": [], "HitEntities": [], "HitGroups": {}}

        final_query = """
                UNWIND $ids AS entity_id
                MATCH (e:__Entity__ {id: entity_id})<-[:MENTIONS]-(c:__Chunk__)
                WITH c, count(distinct e) AS coverage
                RETURN c.id AS id, c.text AS text, c.fileName AS fileName, coverage  // 增加 fileName
                ORDER BY coverage DESC, id ASC
                LIMIT 10
                """

        chunk_res = self.db_query(final_query, {"ids": list(all_hit_entity_ids)})

        return {
            "Chunks": chunk_res.to_dict('records'),
            "HitEntities": list(all_hit_entity_ids)
        }


    def vector_search_by_core(self, core_summary: str) -> list:
        """核心总结词向量检索"""
        final_query = self.retrieval_query.replace("$topChunks", str(self.top_chunks)) \
            .replace("$topCommunities", str(self.top_communities)) \
            .replace("$topOutsideRels", str(self.top_outside_rels)) \
            .replace("$topInsideRels", str(self.top_inside_rels))

        # 初始化向量存储，传入处理后的 final_query
        from langchain_community.vectorstores import Neo4jVector
        vector_store = Neo4jVector.from_existing_index(
            self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=self.index_name,
            text_node_property="description",
            retrieval_query=final_query
        )
        # 返回命中 Top-K 的实体描述及关联 Chunk 内容
        return vector_store.similarity_search(core_summary, k=self.top_entities)

    def vector_search_communities_and_chunks(self, core_summary: str) -> dict:
        """重构后的B路：直接基于总结词检索社区摘要和文本块，带出fileName"""
        emb = self.embeddings.embed_query(core_summary)

        # 检索社区摘要
        comm_res = self.db_query("""
            CALL db.index.vector.queryNodes('community_vector_index', 3, $emb) 
            YIELD node, score 
            RETURN node.summary AS text, node.id AS id, score
        """, {"emb": emb})

        # 检索文本块 (带出 fileName)
        chunk_res = self.db_query("""
            CALL db.index.vector.queryNodes('chunk_vector_index', 5, $emb) 
            YIELD node, score 
            RETURN node.text AS text, node.id AS id, node.fileName AS fileName, score
        """, {"emb": emb})

        return {
            "Communities": comm_res.to_dict('records'),
            "Chunks": chunk_res.to_dict('records')
        }
