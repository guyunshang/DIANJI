import sys
from pathlib import Path

# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

import re
import concurrent.futures
from typing import List, Set, Dict
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from graphrag_agent.graph.core import connection_manager
from graphrag_agent.config.settings import (
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    MAX_WORKERS as DEFAULT_MAX_WORKERS,
    )


class GraphWriter:
    """
    图写入器，负责将提取的实体和关系写入Neo4j图数据库。
    处理实体和关系的解析、转换为GraphDocument，以及批量写入图数据库。
    """

    def __init__(self, graph: Neo4jGraph = None, entity_types: List[str] = None, relationship_types: List[str] = None,
                 relationship_definitions: dict = None,batch_size: int = 50, max_workers: int = 2):
        """
        初始化图写入器
        """
        self.graph = graph or connection_manager.get_connection()
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS

        # 节点缓存，用于减少重复节点的创建
        self.node_cache = {}

        # 用于跟踪已经处理的节点，减少重复操作
        self.processed_nodes: Set[str] = set()

        if entity_types is None:
            # 仅作为兜底，可选
            from graphrag_agent.config.settings import entity_types as global_et
            entity_types = global_et

        self.valid_entity_types = set(entity_types) if entity_types else set()

        if relationship_types is None:
            from graphrag_agent.config.settings import relationship_types as global_rt
            relationship_types = global_rt
        self.valid_relationship_types = set(relationship_types) if relationship_types else set()

        if relationship_definitions is None:
            from graphrag_agent.config.settings import relationship_definitions as global_rd
            relationship_definitions = global_rd
        self.schema = self._parse_relationship_schema(relationship_definitions)

    def _parse_relationship_schema(self, definitions: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
        """
        从自然语言定义中解析关系约束 (源 -> 目标)
        """
        schema = {}
        pattern = re.compile(r'\(源[:：]\s*(.*?)\s*->\s*目标[:：]\s*(.*?)\)')

        for rel_name, desc in definitions.items():
            match = pattern.search(desc)
            if match:
                source_str = match.group(1).strip()
                target_str = match.group(2).strip()
                sources = [s.strip() for s in source_str.split('/')]
                targets = [t.strip() for t in target_str.split('/')]
                schema[rel_name] = {"source": sources, "target": targets}
        return schema

    def convert_to_graph_document(self, chunk_id: str, input_text: str, result: str) -> GraphDocument:
        """
        将提取的实体关系文本转换为GraphDocument对象
        *** 包含Schema过滤与智能补全逻辑 ***
        """
        # 宽松正则 (兼容中文标点、换行、空格)
        # node_pattern = re.compile(
        #     r'(?s)\("entity"\s*[:：]\s*["“](.+?)["”]\s*[:：]\s*["“](.+?)["”]\s*[:：]\s*["“](.+?)["”]\s*\)'
        # )
        # relationship_pattern = re.compile(
        #     r'(?s)\("relationship"\s*[:：]\s*["“](.+?)["”]\s*[:：]\s*["“](.+?)["”]\s*[:：]\s*["“](.+?)["”]\s*[:：]\s*["“](.+?)["”]\s*[:：]\s*(.+?)\s*\)'
        # )
        node_pattern = re.compile(
            r'(?:[(（])\s*["\'“]entity["\'”]\s*[:：]\s*["\'“](.+?)["\'”]\s*[:：]\s*["\'“](.+?)["\'”]\s*[:：]\s*["\'“](.+?)["\'”]\s*(?:[)）])'
        )
        relationship_pattern = re.compile(
            r'(?:[(（])\s*["\'“]relationship["\'”]\s*[:：]\s*["\'“](.+?)["\'”]\s*[:：]\s*["\'“](.+?)["\'”]\s*[:：]\s*["\'“](.+?)["\'”]\s*[:：]\s*["\'“](.+?)["\'”]\s*[:：]\s*([0-9.]+)\s*(?:[)）])'
        )

        nodes = {}
        relationships = []

        # --- 调试打印代码开始 ---
        # print(f"\n{'=' * 50}")
        # print(f"[DEBUG] 正在处理 Chunk ID: {chunk_id}")
        # # 打印 LLM 返回结果的前 300 个字符，方便快速查看格式
        # print(f"[DEBUG] LLM 原始输出 (Top 300 chars):\n{result[:300]}...")
        # if len(result) > 300:
        #     print("...")
        # print(f"{'-' * 50}")
        # --- 调试代码结束 ---

        try:
            # --- 解析节点 ---
            node_matches = node_pattern.findall(result)

            for match in node_matches:
                node_id, node_type, description = match
                node_id = node_id.strip()
                node_type = node_type.strip()
                description = description.strip()

                # === 修改：实体白名单过滤 ===
                if node_type not in self.valid_entity_types:
                    continue

                if node_id in self.node_cache:
                    nodes[node_id] = self.node_cache[node_id]
                elif node_id not in nodes:
                    new_node = Node(
                        id=node_id,
                        type=node_type,
                        properties={'id': node_id, 'description': description}
                    )
                    nodes[node_id] = new_node
                    self.node_cache[node_id] = new_node

            # --- 解析关系 ---
            rel_matches = relationship_pattern.findall(result)

            for match in rel_matches:
                source_id, target_id, rel_type, description, weight = match
                source_id = source_id.strip()
                target_id = target_id.strip()
                rel_type = rel_type.strip()
                description = description.strip()

                # === 修改：关系白名单过滤 ===
                if rel_type not in self.valid_relationship_types:
                    continue

                # 获取该关系的Schema约束
                schema_rule = self.schema.get(rel_type)

                # === 修改：基于Schema的智能补全 (源节点) ===
                if source_id not in nodes:
                    # 检查是否有缓存
                    if source_id in self.node_cache:
                        nodes[source_id] = self.node_cache[source_id]
                    # 尝试推断补全
                    # elif schema_rule and schema_rule.get("source"):
                    #     inferred_type = schema_rule["source"][0]  # 取第一个允许的类型
                    #     if inferred_type in self.valid_entity_types:
                    #         new_node = Node(
                    #             id=source_id,
                    #             type=inferred_type,
                    #             properties={
                    #                 'id': source_id,
                    #                 'description': f'根据关系[{rel_type}]自动推断补全',
                    #                 'is_inferred': True
                    #             }
                    #         )
                    #         nodes[source_id] = new_node
                    #         self.node_cache[source_id] = new_node
                    #       else:
                    #           continue  # 推断类型非法，丢弃
                    else:
                        continue  # 无法推断，丢弃关系

                # === 修改：基于Schema的智能补全 (目标节点) ===
                if target_id not in nodes:
                    if target_id in self.node_cache:
                        nodes[target_id] = self.node_cache[target_id]
                    # elif schema_rule and schema_rule.get("target"):
                    #     inferred_type = schema_rule["target"][0]
                    #     if inferred_type in self.valid_entity_types:
                    #         new_node = Node(
                    #             id=target_id,
                    #             type=inferred_type,
                    #             properties={
                    #                 'id': target_id,
                    #                 'description': f'根据关系[{rel_type}]自动推断补全',
                    #                 'is_inferred': True
                    #             }
                    #         )
                    #         nodes[target_id] = new_node
                    #         self.node_cache[target_id] = new_node
                    #       else:
                    #           continue
                    else:
                        continue

                # 处理权重
                try:
                    weight_clean = re.sub(r'[^\d\.]', '', str(weight))
                    weight_val = float(weight_clean) if weight_clean else 1.0
                except:
                    weight_val = 1.0

                relationships.append(
                    Relationship(
                        source=nodes[source_id],
                        target=nodes[target_id],
                        type=rel_type,
                        properties={
                            "description": description,
                            "weight": weight_val
                        }
                    )
                )

        except Exception as e:
            print(f"[ERROR] 解析 Chunk {chunk_id} 时发生严重错误: {str(e)}")
            return GraphDocument(
                nodes=[],
                relationships=[],
                source=Document(
                    page_content=input_text,
                    metadata={"chunk_id": chunk_id, "error": str(e)}
                )
            )

        return GraphDocument(
            nodes=list(nodes.values()),
            relationships=relationships,
            source=Document(
                page_content=input_text,
                metadata={"chunk_id": chunk_id}
            )
        )

    def process_and_write_graph_documents(self, file_contents: List) -> None:
        """
        处理并写入所有文件的GraphDocument对象 - 使用并行处理和批处理优化

        Args:
            file_contents: 文件内容列表
        """
        all_graph_documents = []
        all_chunk_ids = []

        # 预分配列表大小
        total_chunks = sum(len(file_content[3]) for file_content in file_contents)
        all_graph_documents = [None] * total_chunks
        all_chunk_ids = [None] * total_chunks

        chunk_index = 0
        error_count = 0

        print(f"开始处理 {total_chunks} 个chunks的GraphDocument")

        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {}

            # 提交所有任务
            for file_content in file_contents:
                chunks = file_content[3]  # chunks_with_hash在索引3的位置
                results = file_content[4]  # 提取结果在索引4的位置

                for i, (chunk, result) in enumerate(zip(chunks, results)):
                    future = executor.submit(
                        self.convert_to_graph_document,
                        chunk["chunk_id"],
                        chunk["chunk_doc"].page_content,
                        result
                    )
                    future_to_index[future] = chunk_index
                    chunk_index += 1

            # 收集处理结果
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    graph_document = future.result()

                    # 只保留有效的图文档
                    if len(graph_document.nodes) > 0 or len(graph_document.relationships) > 0:
                        all_graph_documents[idx] = graph_document
                        all_chunk_ids[idx] = graph_document.source.metadata.get("chunk_id")
                    else:
                        all_graph_documents[idx] = None
                        all_chunk_ids[idx] = None

                except Exception as e:
                    error_count += 1
                    print(f"处理chunk时出错 (已有{error_count}个错误): {e}")
                    all_graph_documents[idx] = None
                    all_chunk_ids[idx] = None

        # 过滤掉None值
        all_graph_documents = [doc for doc in all_graph_documents if doc is not None]
        all_chunk_ids = [id for id in all_chunk_ids if id is not None]

        print(f"共处理 {total_chunks} 个chunks, 有效文档 {len(all_graph_documents)}, 错误 {error_count}")

        # 批量写入图文档
        self._batch_write_graph_documents(all_graph_documents)

        # 批量合并chunk关系
        if all_chunk_ids:
            self.merge_chunk_relationships(all_chunk_ids)

    def _batch_write_graph_documents(self, documents: List[GraphDocument]) -> None:
        """
        批量写入图文档

        Args:
            documents: 图文档列表
        """
        if not documents:
            return

        # 增加批处理大小的动态调整
        optimal_batch_size = min(self.batch_size, max(10, len(documents) // 10))
        total_batches = (len(documents) + optimal_batch_size - 1) // optimal_batch_size

        print(f"开始批量写入 {len(documents)} 个文档，批次大小: {optimal_batch_size}, 总批次: {total_batches}")

        # 批量写入图文档
        for i in range(0, len(documents), optimal_batch_size):
            batch = documents[i:i + optimal_batch_size]
            if batch:
                try:
                    self.graph.add_graph_documents(
                        batch,
                        baseEntityLabel=True,
                        include_source=True
                    )
                    print(f"已写入批次 {i // optimal_batch_size + 1}/{total_batches}")
                except Exception as e:
                    print(f"写入图文档批次时出错: {e}")
                    # 如果批次写入失败，尝试逐个写入以避免整批失败
                    for doc in batch:
                        try:
                            self.graph.add_graph_documents(
                                [doc],
                                baseEntityLabel=True,
                                include_source=True
                            )
                        except Exception as e2:
                            print(f"单个文档写入失败: {e2}")

    def merge_chunk_relationships(self, chunk_ids: List[str]) -> None:
        """
        合并Chunk节点与Document节点的关系

        Args:
            chunk_ids: 块ID列表
        """
        if not chunk_ids:
            return

        # 去除重复的chunk_id以减少操作数量
        unique_chunk_ids = list(set(chunk_ids))
        print(f"开始合并 {len(unique_chunk_ids)} 个唯一chunk关系")

        # 动态批处理大小
        optimal_batch_size = min(self.batch_size, max(20, len(unique_chunk_ids) // 5))
        total_batches = (len(unique_chunk_ids) + optimal_batch_size - 1) // optimal_batch_size

        print(f"合并关系批次大小: {optimal_batch_size}, 总批次: {total_batches}")

        # 分批处理，避免一次性处理过多数据
        for i in range(0, len(unique_chunk_ids), optimal_batch_size):
            batch_chunk_ids = unique_chunk_ids[i:i + optimal_batch_size]
            batch_data = [{"chunk_id": chunk_id} for chunk_id in batch_chunk_ids]

            try:
                # 使用原始的查询，确保兼容性
                merge_query = """
                    UNWIND $batch_data AS data
                    MATCH (c:`__Chunk__` {id: data.chunk_id}), (d:Document{chunk_id:data.chunk_id})
                    WITH c, d
                    MATCH (d)-[r:MENTIONS]->(e)
                    MERGE (c)-[newR:MENTIONS]->(e)
                    ON CREATE SET newR += properties(r)
                    DETACH DELETE d
                """

                self.graph.query(merge_query, params={"batch_data": batch_data})
                print(f"已处理合并关系批次 {i // optimal_batch_size + 1}/{total_batches}")
            except Exception as e:
                print(f"合并关系批次时出错: {e}")
                # 如果批处理失败，尝试逐个处理
                for chunk_id in batch_chunk_ids:
                    try:
                        single_query = """
                            MATCH (c:`__Chunk__` {id: $chunk_id}), (d:Document{chunk_id:$chunk_id})
                            WITH c, d
                            MATCH (d)-[r:MENTIONS]->(e)
                            MERGE (c)-[newR:MENTIONS]->(e)
                            ON CREATE SET newR += properties(r)
                            DETACH DELETE d
                        """
                        self.graph.query(single_query, params={"chunk_id": chunk_id})
                    except Exception as e2:
                        print(f"处理单个chunk关系时出错: {e2}")