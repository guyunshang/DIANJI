import sys
from pathlib import Path
from unittest.mock import MagicMock

# 1. 确保路径正确
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# 2. 导入配置（这里会读取你修改后的 settings.py）
from langchain_core.runnables import RunnableLambda
from graphrag_agent.config.settings import (
    entity_types,
    relationship_types,
    entity_definitions,  # 检查这里是否能导入
    relationship_definitions  # 检查这里是否能导入
)
from graphrag_agent.config.prompts.graph_prompts import system_template_build_graph, human_template_build_graph
from graphrag_agent.graph.extraction.entity_extractor import EntityRelationExtractor


def mock_llm_invoke(input_data):
    return MagicMock(content="mock_result")


def test_success_case():
    print("=== 开始验证修改后的逻辑 ===")

    mock_llm = RunnableLambda(mock_llm_invoke)

    # 3. 模拟【修复后】的 build_graph.py 初始化方式
    # 注意：这里我们显式传入了定义的字典
    extractor = EntityRelationExtractor(
        llm=mock_llm,
        system_template=system_template_build_graph,
        human_template=human_template_build_graph,
        entity_types=entity_types,
        relationship_types=relationship_types,
        entity_definitions=entity_definitions,  # 传入定义
        relationship_definitions=relationship_definitions,  # 传入定义
        max_workers=4,
        batch_size=5
    )

    print("\n--- 传入大模型的实体定义内容 ---")
    print(extractor.formatted_entity_types)
    print(extractor.formatted_relationship_types)

    # 4. 判定逻辑
    if "无详细定义" in extractor.formatted_entity_types:
        print("\n[结果]: ❌ 仍然失败！")
        print("原因可能是：")
        print("1. 你的 settings.py 中 entity_definitions 字典里缺某些 Key。")
        print(f"   当前 extractor 中的类型有: {entity_types}")
        print(f"   settings.py 中的定义 Key 有: {list(entity_definitions.keys())}")
    else:
        print("\n[结果]: ✅ 验证成功！现在大模型可以读取到你的详细定义了。")


if __name__ == "__main__":
    test_success_case()