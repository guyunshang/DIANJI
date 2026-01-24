import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

"""
Executor层Prompt模板集合

包含任务执行、反思和重新规划的Prompt模板
"""

# 功能: 执行单个子任务
EXECUTE_PROMPT = '''你是一个任务执行助手。你需要执行计划中的一个具体子任务。

**完整计划概览**:
{plan_summary}

**当前任务**:
- 任务ID: {task_id}
- 任务类型: {task_type}
- 任务描述: {task_description}
- 相关实体: {entities}

**已完成的前置任务**:
{past_steps}

**可用工具**:
{tools_description}

**执行要求**:
1. 严格按照任务描述执行，不要偏离目标
2. 充分利用已完成任务的结果（如有依赖）
3. 合理选择和调用工具
4. 收集充分的证据支持结论
5. 如果遇到问题，如实汇报而不是编造信息

**不同任务类型的执行策略**:

**local_search** (精确检索):
- 使用LocalSearchTool在指定实体所属社区内检索
- 关注具体细节和明确关系
- 示例: 检索"孙悟空的师父" -> 在"人物"社区检索"孙悟空"的"师徒"关系

**global_search** (全局概览):
- 使用GlobalSearchTool获取跨社区的主题摘要
- 适合理解整体概念和背景
- 示例: "悟空传的主题" -> 汇总所有社区的摘要信息

**deep_research** (深度研究):
- 使用DeeperResearchTool进行多轮推理
- 结合ThinkingEngine生成思考过程
- 构建证据链支持复杂结论
- 示例: "分析孙悟空反抗天庭的原因" -> 多轮检索和推理

**chain_exploration** (路径探索):
- 使用ChainOfExplorationTool追踪实体间关系路径
- 从起始实体出发，沿关系链探索目标
- 示例: "孙悟空如何影响唐僧的决定" -> 追踪"孙悟空→事件→唐僧"路径

现在请执行任务并返回结果。

**执行结果**（必须输出合法JSON，字段含义如下）:
```json
{
  "task_id": "{task_id}",
  "status": "success | partial | failed",
  "summary": "对执行过程和结论的1-3段描述",
  "actions": [
    {
      "tool": "使用的工具名称",
      "input": "调用时的核心输入",
      "output": "工具返回的要点"
    }
  ],
  "evidence": [
    {
      "result_id": "RetrievalResult.result_id",
      "granularity": "RetrievalResult.granularity",
      "source": "RetrievalResult.source",
      "confidence": 0.0,
      "citation": "可直接引用的文本或说明"
    }
  ],
  "notes": "遇到的限制、下一步建议等（如无可留空字符串）"
}
```
确保 `evidence` 数组中的每一项都能映射到可存入 `RetrievalResult` 的字段。'''


# 功能: 对执行结果进行反思和评估
REFLECT_PROMPT = '''你是一个质量评估助手。你需要对任务执行结果进行反思和评估。

**任务信息**:
- 任务ID: {task_id}
- 任务描述: {task_description}
- 任务类型: {task_type}

**执行结果**:
{execution_result}

**收集的证据** ({evidence_count}条):
{evidence_summary}

**评估标准**:
1. **完整性**: 任务目标是否完全达成？
2. **证据充分性**: 证据数量和质量是否足够支持结论？
3. **一致性**: 不同证据之间是否有冲突？
4. **相关性**: 检索的内容是否与任务描述相关？
5. **置信度**: 对结果的总体信心如何？

**判断逻辑**:
- **成功**: 完整达成目标，证据充分且一致，置信度 > 0.7
- **部分成功**: 基本达成目标，但证据不够充分或存在小问题，置信度 0.5-0.7
- **失败**: 未能达成目标，证据不足或冲突严重，置信度 < 0.5

**示例1 - 成功**:
任务: "检索变压器相间短路故障的表现"
结果: 找到5条证据，均与原文一致，没有错缺
评估:
```json
{{
  "success": true,
  "confidence": 0.95,
  "suggestions": [],
  "needs_retry": false,
  "reasoning": "任务完全达成，证据充分且一致，信息明确"
}}
```

**示例2 - 需要改进**:
任务: "分析如何应对电缆护套破损"
结果: 仅找到“仅护套破损”的抢修方法与预防措施，缺少对由于护套破损导致的电缆进水应对的深层分析
评估:
```json
{{
  "success": false,
  "confidence": 0.5,
  "suggestions": [
    "增加对电缆进水故障的分析，包括抢修方法与预防措施",
    "思考进水程度严重与否应对方法是否一致",
    "与原文内容是否一致",
  ],
  "needs_retry": true,
  "reasoning": "结果不全面，还需进一步比对原文分析"
}}
```

现在请对以上执行结果进行反思评估，严格按照JSON格式输出。引用证据时请使用`result_id`（如 `[result_id]`）。

**反思结果**:
```json
'''


# 功能: 基于执行结果重新规划，或生成最终响应
REPLAN_PROMPT = '''你是一个规划调整助手。根据当前执行情况，你需要决定是否需要调整计划，或者已经可以生成最终答案。

**原始目标**: {goal}

**原始计划**:
{original_plan}

**已执行任务及结果**:
{execution_records}

**当前状态**:
- 已完成任务数: {completed_count} / {total_count}
- 累计token消耗: {token_used}
- 收集的证据总数: {total_evidence}

**决策逻辑**:

**情况1: 可以生成最终答案** (满足以下条件之一):
- 所有任务已完成且验收标准已满足
- 已收集足够证据可以回答用户问题（即使有任务未执行）
- Token预算即将耗尽，需要基于现有信息作答

此时返回 Response类型，包含最终答案。

**情况2: 需要调整计划** (满足以下条件之一):
- 某些任务执行失败或结果不理想
- 发现新的重要问题需要探索
- 任务依赖关系需要调整
- 需要增加/删除/修改任务

此时返回 Plan类型，包含调整后的任务列表。

**示例1 - 生成最终答案**:
原始问题: "断路器灭弧室故障的抢修首先要做什么?"
已完成: task_001 (local_search) -> 找到确定答案"安全隔离"，证据充分
决策: 可以直接回答

输出:
```json
{{
  "action_type": "Response",
  "response": "根据检索结果，断路器灭弧室故障的抢修首先要安全隔离：将故障断路器两侧可靠隔离，合上接地刀闸或挂设接地线，并释放操作机构的能量。\\n\\n【证据来源】\\n1. [chunk_123] 断路器故障分类详述: \\"抢修步骤：第一步是安全隔离：将故障断路器两侧可靠隔离，合上接地刀闸或挂设接地线，并释放操作机构的能量。...\\"\\n2. [entity_456] 知识图谱记录: 灭弧室故障 -抢修方法-> 安全隔离\\n\\n断路器灭弧室故障抢修的第一步室安全隔离。"
}}
```

**示例2 - 调整计划**:
原始问题: "电缆主绝缘击穿故障的分类由什么决定"
已完成: task_001 (local_search 高阻故障、低阻故障) -> 仅获得分类情况，深层原因不明
当前问题: 缺少对分类原因的分析

输出:
```json
{{
  "action_type": "Plan",
  "updated_plan": {{
    "nodes": [
      {{
        "task_id": "task_003",
        "task_type": "chain_exploration",
        "description": "追踪主绝缘击穿故障的因果链",
        "priority": 1,
        "estimated_tokens": 800,
        "depends_on": ["task_001", "task_002"],
        "entities": ["主绝缘击穿故障", "高阻故障", "低阻故障"],
        "status": "pending"
      }},
      {{
        "task_id": "task_004",
        "task_type": "deep_research",
        "description": "深度分析分类的根据（击穿电阻等维度）",
        "priority": 1,
        "estimated_tokens": 1500,
        "depends_on": ["task_003"],
        "status": "pending"
      }}
    ],
    "execution_mode": "sequential"
  }},
  "reasoning": "当前证据仅获得分类情况，深层原因不明，需要增加chain_exploration追踪原因链，再通过deep_research进行深层分析"
}}
```

现在请根据当前执行状态做出决策，严格按照JSON格式输出：

**决策结果**:
```json
'''
