import sys
import os
from pathlib import Path

# 1. 环境初始化
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from graphrag_agent.models.get_models import get_embeddings_model
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import LOCAL_SEARCH_SETTINGS, similarity_threshold
from langchain_community.vectorstores import Neo4jVector


def run_vector_debug():
    query = "电缆主绝缘击穿故障的表现及预防"
    print(f"=== 开始向量检索底层调试 ===")
    print(f"原始问题: {query}")
    print(f"系统相似度阈值: {similarity_threshold}")

    # 2. 检查 Embedding 模型
    try:
        embeddings = get_embeddings_model()
        test_vec = embeddings.embed_query("测试文本")
        print(f"✅ Embedding 模型加载成功，向量维度: {len(test_vec)}")
    except Exception as e:
        print(f"❌ Embedding 模型报错: {e}")
        return

    # 3. 检查 Neo4j 连接与索引
    db_manager = get_db_manager()
    index_name = LOCAL_SEARCH_SETTINGS["index_name"]
    print(f"目标向量索引名: {index_name}")

    # 4. 直接执行相似度搜索（查看原始得分）
    try:
        vector_store = Neo4jVector.from_existing_index(
            embeddings,
            url=db_manager.neo4j_uri,
            username=db_manager.neo4j_username,
            password=db_manager.neo4j_password,
            index_name=index_name,
            text_node_property = "description"
        )

        print("\n--- 环节 1：纯向量匹配测试 (top_k=5) ---")
        # 使用 search_with_score 获取原始相似度
        results_with_score = vector_store.similarity_search_with_score(query, k=5)

        if not results_with_score:
            print("⚠️ 警告：向量库未返回任何结果！可能是索引为空或未同步。")
        else:
            for i, (doc, score) in enumerate(results_with_score):
                print(f"[{i + 1}] 相似度分数: {score:.4f}")
                print(f"    内容摘要: {doc.page_content}...")
                # 如果分数值 < similarity_threshold，系统会自动丢弃该结果
                if score < similarity_threshold:
                    print(f"    ❌ 该结果低于系统阈值 ({similarity_threshold})，将被忽略。")
    except Exception as e:
        print(f"❌ 向量检索执行失败: {e}")

    # 5. 检查图谱拓扑关系
    print("\n--- 环节 2：图谱拓扑关系检查 ---")
    driver = db_manager.get_driver()
    with driver.session() as session:
        # A. 检查 Chunk 总数
        count_res = session.run("MATCH (c:__Chunk__) RETURN count(c) as count").single()
        print(f"库中 __Chunk__ 节点总数: {count_res['count']}")

        # B. 检查 MENTIONS 关系 (这是 Local Search 能够关联到 Chunk 的核心)
        rel_res = session.run("MATCH ()-[:MENTIONS]->() RETURN count(*) as count").single()
        print(f"库中 [:MENTIONS] 关系总数: {rel_res['count']}")

        # C. 验证特定关键词是否建立了链接
        # 这里的 '电缆' 应替换为您确定存在的实体名
        test_entity = "电缆"
        link_check = session.run(
            "MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__) "
            # 【必须修改】去掉 e.name，只保留 e.id，因为库里没有 name 属性
            "WHERE e.id = $eid "
            "RETURN c.text LIMIT 1", {"eid": test_entity}
        ).single()

        if link_check:
            print(f"✅ 实体 '{test_entity}' 已成功关联到文本块。")
        else:
            print(f"❌ 错误：实体 '{test_entity}' 没有任何关联的文本块。Local Search 必定失效！")


if __name__ == "__main__":
    run_vector_debug()