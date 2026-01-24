import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

"""
Planner层Prompt模板集合

包含任务分解、澄清和计划审校的Prompt模板
"""

# 功能: 将用户查询分解为可执行的子任务，生成TaskGraph
TASK_DECOMPOSE_PROMPT = '''你是一个专业的任务规划助手。你的职责是将用户的复杂查询分解为清晰、可执行的子任务序列。

**用户查询**: {query}

**最大任务数**: {max_tasks}

**可用任务类型**:
1. **local_search**: 在知识图谱中检索特定实体的详细信息和局部关系（微观视角，针对具体实体）
2. **global_search**: 在知识图谱中检索整体概念和社区级摘要信息（宏观视角，针对主题概念）
3. **hybrid_search**: 结合图谱结构和向量语义的综合检索，适合需要宏微观结合的场景
4. **naive_search**: 直接向量检索，快速获取原文片段或概念解释
5. **deep_research** / **deeper_research**: 深度研究和多轮推理，构建完整证据链或复杂分析
6. **chain_exploration**: 图谱路径探索，通过实体关系链追踪信息，适合"如何达成"类问题
7. **reflection**: 对已完成任务进行质量校验或补充改进建议，通常依赖已有答案

**分解原则**:
1. 每个子任务应该是独立、原子化的操作
2. 任务之间可以有依赖关系，但不要过度依赖
3. 优先级分配: 1(高优先级，基础性任务) 2(中优先级) 3(低优先级，增强性任务)
4. 预估token消耗要合理（简单查询300-500，复杂查询500-1000，深度研究1000-2000）
5. 避免创建冗余任务
6. 初始化每个任务的状态为 "pending"

**示例1 - 简单查询**:
查询: "电缆的故障类型有哪些?"
分解结果:
```json
{{
  "nodes": [
    {{
      "task_id": "task_001",
      "task_type": "local_search",
      "description": "在知识图谱中检索电缆的故障类型",
      "priority": 1,
      "estimated_tokens": 400,
      "depends_on": [],
      "entities": ["电缆"],
      "status": "pending"
    }}
  ],
  "execution_mode": "sequential"
}}
```

**示例2 - 复杂查询**:
查询: "分析电缆护套破损与进水故障的抢修方法"
分解结果:
```json
{{
  "nodes": [
    {{
      "task_id": "task_001",
      "task_type": "local_search",
      "description": "检索电缆护套破损的抢修方法",
      "priority": 1,
      "estimated_tokens": 500,
      "depends_on": [],
      "entities": ["电缆", "护套破损"],
      "status": "pending"
    }},
    {{
      "task_id": "task_002",
      "task_type": "local_search",
      "description": "检索电缆进水故障的抢修方法",
      "priority": 1,
      "estimated_tokens": 500,
      "depends_on": [],
      "entities": ["电缆", "进水故障"],
      "status": "pending"
    }},
    {{
      "task_id": "task_003",
      "task_type": "chain_exploration",
      "description": "辨别电缆护套破损与进水故障的区别",
      "priority": 2,
      "estimated_tokens": 800,
      "depends_on": ["task_001", "task_002"],
      "entities": ["孙悟空", "天庭"],
      "status": "pending"
    }},
    {{
      "task_id": "task_004",
      "task_type": "deep_research",
      "description": "深度分析电缆护套破损与进水故障的抢修逻辑",
      "priority": 2,
      "estimated_tokens": 1200,
      "depends_on": ["task_003"],
      "status": "pending"
    }},
    {{
      "task_id": "task_005",
      "task_type": "global_search",
      "description": "获取电缆护套破损与进水故障的抢修步骤详述",
      "priority": 3,
      "estimated_tokens": 600,
      "depends_on": [],
      "status": "pending"
    }}
  ],
  "execution_mode": "adaptive"
}}
```

现在请针对以下查询生成任务分解方案，严格按照JSON格式输出：

**查询**: {query}

**任务分解方案**:
```json
'''


# 功能: 检测查询模糊性，生成澄清问题
CLARIFY_PROMPT = '''你是一个查询澄清助手。你的职责是识别用户查询中的模糊性和不确定性，并生成针对性的澄清问题。

**用户查询**: {query}

**领域背景**: {domain}

**需要检查的模糊性类型**:
1. **领域范围不明确**: 查询涉及多个可能的领域或主题
2. **时间范围缺失**: 查询涉及时间信息但未指定具体时期
3. **粒度不清晰**: 用户需要概览还是详细信息不明确
4. **实体歧义**: 提到的实体名称可能有多种含义
5. **意图不明**: 用户是想知道"是什么"、"为什么"还是"如何做"

**判断标准**:
- 如果查询清晰明确，无需澄清，返回 `needs_clarification: false`
- 如果存在以上任一模糊性，返回 `needs_clarification: true` 并列出具体问题

**示例1 - 需要澄清**:
查询: "避雷器故障处理"
分析: 存在多重模糊性
- 实体歧义: "避雷器"可能指10kV线路避雷器、变电站阀式避雷器或氧化锌避雷器
- 粒度不清: 是指所有故障类型的处理，还是针对特定现象（如爆炸、闪络）的处理
- 意图不明: 是指现场抢修的操作步骤，还是事故后的管理补救措施

输出:
```json
{{
  "needs_clarification": true,
  "questions": [
    "您是指哪类设备的避雷器？(1)10kV配电线路避雷器 (2)变电站主变避雷器",
    "您是想了解哪种具体的故障现象？例如受潮、闪络还是阀片老化？",
    "您需要的是现场抢修操作指导，还是预防性维护的措施建议？"
  ],
  "ambiguity_types": ["实体歧义", "粒度不清晰", "意图不明"]
}}
```

**示例2 - 无需澄清**:
查询: "10kV线路氧化锌避雷器由于受潮导致的闪络故障应如何处理?" 分析: 查询明确清晰
- 领域明确: 10kV线路/电力设备运维
- 实体明确: 氧化锌避雷器
- 故障明确: 受潮导致的闪络
- 意图明确: 寻找处理方法


输出:
```json
{{
  "needs_clarification": false,
  "questions": [],
  "ambiguity_types": []
}}
```

现在请分析以下查询，严格按照JSON格式输出：

**查询**: {query}

**澄清分析结果**:
```json
'''


# 功能: 审校TaskGraph，生成完整的PlanSpec
PLAN_REVIEW_PROMPT = '''你是一个计划审校助手。你的职责是审核任务分解方案(TaskGraph)，确保其合理性和可执行性，并生成完整的执行计划(PlanSpec)。

**用户原始查询**: {query}

**重写后的查询**: {refined_query}

**任务图(TaskGraph)**:
```json
{task_graph}
```

**用户确认的假设条件**: {assumptions}

**审校检查项**:
1. **任务数量**: 是否在合理范围内（建议1-8个任务）
2. **依赖关系**: 是否存在循环依赖或无法满足的依赖
3. **任务类型匹配**: 每个任务的类型是否适合其描述的操作
4. **优先级合理性**: 基础任务应该是高优先级
5. **Token预估**: 总token消耗是否在预算内（建议 < 10000）
6. **执行模式**: sequential(简单查询) vs adaptive(复杂查询) 的选择是否恰当
7. **状态字段**: 每个任务的初始 `status` 应为 `"pending"`

**生成内容**:
1. **问题陈述(ProblemStatement)**: 清晰描述用户要解决的问题
2. **验收标准(AcceptanceCriteria)**: 定义任务完成的标准
3. **审校意见**: 指出潜在问题并给出修改建议

**示例输出**:
```json
{{
  "problem_statement": {{
    "original_query": "分析10kV避雷器的故障原因",
    "refined_query": "综合分析10kV线路避雷器在运行过程中出现受潮、老化及闪络的深层诱因及对应的故障表现",
    "background_info": "该知识图谱包含电气设备、故障类型、故障原因及抢修方法等本体定义",
    "user_intent": "诊断设备健康状态并提供运维支持"
  }},
  "acceptance_criteria": {{
    "completion_conditions": [
      "涵盖10kV避雷器的主要故障模式（如受潮、密封不良）",
      "关联出故障表现与故障原因之间的逻辑关系",
      "提供图谱中关于预防措施的证据"
    ],
    "quality_requirements": [
      "必须引用具体的故障分类数据",
      "逻辑链条需满足：设备->表现->原因",
      "避免非电力专业的通用性推断"
    ],
    "min_evidence_count": 4,
    "min_confidence": 0.8
  }},
  "validation_results": {{
    "is_valid": true,
    "issues": [],
    "suggestions": [
      "建议在task_002之后增加一个global_search任务，获取配网避雷器运行管理的整体趋势摘要"
    ],
    "estimated_total_tokens": 2500,
    "estimated_time_minutes": 3
  }}
}}
```

现在请审校以上任务图并生成完整计划，严格按照JSON格式输出：

**PlanSpec**:
```json
'''
