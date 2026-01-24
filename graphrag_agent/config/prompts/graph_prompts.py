import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

"""
图谱构建与社区摘要提示模板集合。

这些模板用于图谱索引的构建与维护流程。
"""

system_template_build_graph = """
-目标- 
你是一个专业的知识图谱构建助手。你的任务是根据给定的本体模型，从文本中精准提取实体和关系。该知识图谱是描述电气设备故障及处理的知识图谱，旨在建立从电气设备-故障-故障原因/抢修方法/预防措施，以及统计抢修方法中的抢修资源。
 -本体定义-
以下是系统中定义的实体类型名称及其含义，请严格理解这些定义：
{entity_types}
以下是允许的关系类型名称、含义及其约束（注意箭头方向）：
{relationship_types}


-步骤- 
1.识别文本中的实体与关系，确保它们属于已定义实体类型与关系类型。对于每个已识别的实体和关系，提取以下信息： 
实体：
-entity_name：实体名称 
-entity_type：属于的实体类型,不属于已定义类型的实体不抽取
-entity_description：对实体属性和活动的综合描述，必须生成，不能为空
将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

关系：
-source_entity：关系中源实体的名称， 
-target_entity：关系中目标实体的名称
-relationship_type：属于的关系类型，不属于已定义类型的关系不抽取
-relationship_description：解释为什么你认为源实体和目标实体是相互关联的，必须生成，不能为空
-relationship_strength：一个数字评分，表示源实体和目标实体之间关系的强度 
将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>) 

2.实体和关系的所有属性用中文输出，步骤1中识别的所有实体和关系输出为一个列表。使用**{record_delimiter}**作为列表分隔符。 
3.完成后，输出{completion_delimiter}

###################### 
-示例-

Entity_types: 
- 电气设备: 只包括以下设备：变压器、电缆、杆塔、输电线路、避雷器、光缆、环网柜、断路器、断路器，其他不是上述之一的一概禁止抽取为此类型，
- 故障类型: 电气设备可能发生的故障的类型或名称，如匝间短路故障。
- 故障表现: 某种故障类型对应的具体异常状态或现象，如变压器匝间短路时对应的油温异常升高、乙炔含量增长等。
- 故障原因: 导致电气设备某种故障类型或故障表现发生的直接或间接诱因。如变压器的短路匝环流是导致油温异常升高的故障原因
- 抢修人员: 可以是抢修团队、团队中的个人，也可以是其他参与抢修的人员。注意指的是比较明确的对象，如某班组、某人此类具体的应该抽取；如试验人员、运行人员、操作人员这些比较宽泛的禁止抽取
- 抢修资源: 抢修过程中涉及到的工具、材料、仪器仪表或安全用具，如吊车、真空泵等。注意“吊车、真空泵”应视为两个抢修资源，不要因为连接符号而视为一个抢修资源。另外人员、方法（如低压脉冲法）等都不是资源。
- 抢修方法: 针对某种故障抢修要采取的处理步骤、维修手段或操作流程。注意指的是抢修流程或方法，抢修资源禁止归为此类。
- 子步骤: 抢修方法对应的具体操作细节或流程
- 预防措施: 为了防止故障再次发生或减少故障概率而采取的管理或技术措施。
- 情况分支点: 需要分情况或者分类时的点，如<电缆主绝缘击穿故障可分为高阻故障与低阻故障>，则高阻故障与低阻故障为情况分支点。如<对于金属护套破损，使用电缆护层故障测试仪定位破损点，对于疑似进水的电缆，则需在开挖后剖开外护套检查潮气痕迹>，则金属护套破损与疑似进水为情况分支点。

Relationship_types: 
- 故障分类: 描述电气设备存在或出现了某种故障类型。 (源: 电气设备 -> 目标: 故障类型)
- 表现为: 描述电气设备的某种故障对应的具体故障现象或特征。优先描述故障类型对应故障表现，若无故障类型，则描述电气设备对应故障表现。 (源: 故障类型/电气设备 -> 目标: 故障表现)
- 原因为: 描述导致电气设备某种故障类型或故障表现的具体原因。 (源: 故障类型/故障表现 -> 目标: 故障原因)
- 分配维修: 描述团队或人员被指派负责维修某设备或处理某故障。 (源: 抢修人员 -> 目标: 故障类型)
- 需要资源: 描述抢修某故障所需要的物资和工具。 (源: 抢修方法 -> 目标:抢修资源)
- 需执行: 描述解决特定的故障类型或故障表现需要执行某些方法或措施，步骤较多则分步骤书写。优先描述故障类型对应抢修方法，若无故障类型，则描述故障表现对应抢修方法。 (源: 故障类型/故障表现 -> 目标: 抢修方法)
- 负责: 描述某些或某个抢修人员所负责的/执行的抢修方法。 (源: 抢修人员 -> 目标: 抢修方法)
- 下一步: 在抢修方法中有明显的逻辑次序且步骤较多则进行分步骤描述，表示抢修方法的次序关系。 (源: 抢修方法 -> 源: 抢修方法)
- 细节为: 描述抢修方法的具体操作，如快速隔离与初步测距的具体细节为隔离故障电缆两端并验电接地、脉冲法进行波形分析。 (源: 抢修方法 -> 源: 子步骤)
- 可预防: 描述措施可以预防某种故障。 (源: 预防措施 -> 目标: 故障类型/故障表现/故障原因)
- 属于情况: 描述分类或者分情况的情形。 (源: 故障类型/故障表现/抢修方法 -> 目标:情况分支点)

Text:
1.电缆主绝缘击穿故障
现象与特征：这是电缆最常见且最严重的故障，通常由绝缘老化等原因引发。故障发生时，电气试验表现为绝缘电阻为零或极低。根据击穿电阻高低，可分为低阻故障（电阻小于100欧姆）和高阻故障（电阻可达兆欧级）。
抢修步骤：第一步是地面精确定点：根据粗测距离，在电缆路径上使用声磁同步定点仪（监听放电声与磁场信号）或跨步电压法（针对金属护层接地故障）进行精确定位。第二步是开挖与修复：协调资源进行开挖，切除故障段。
预防措施：严格执行电缆敷设规程，避免弯曲半径过小和机械拖伤。
Output:
("entity"{tuple_delimiter}"电缆"{tuple_delimiter}"电气设备"{tuple_delimiter}"电力系统中的主要电气设备，出现主绝缘击穿故障"){record_delimiter}
("entity"{tuple_delimiter}"主绝缘击穿故障"{tuple_delimiter}"故障类型"{tuple_delimiter}"电缆最常见且最严重的故障类型"){record_delimiter}
("entity"{tuple_delimiter}"绝缘电阻为零或极低"{tuple_delimiter}"故障表现"{tuple_delimiter}"电缆发生主绝缘击穿故障时，电气试验表现为绝缘电阻为零或极低"){record_delimiter}
("entity"{tuple_delimiter}"绝缘老化"{tuple_delimiter}"故障原因"{tuple_delimiter}"可能导致电缆主绝缘击穿故障的原因之一"){record_delimiter}
("entity"{tuple_delimiter}"声磁同步定点仪"{tuple_delimiter}"抢修资源"{tuple_delimiter}"用于监听放电声与磁场信号进行精确定位的仪器"){record_delimiter}
("entity"{tuple_delimiter}"地面精确定点"{tuple_delimiter}"抢修方法"{tuple_delimiter}"抢修电缆主绝缘击穿故障的第一步：在电缆路径上进行精确定位"){record_delimiter}
("entity"{tuple_delimiter}"开挖与修复"{tuple_delimiter}"抢修方法"{tuple_delimiter}"抢修电缆主绝缘击穿故障的第二步：开挖故障段并进行修复"){record_delimiter}
("entity"{tuple_delimiter}"使用声磁同步定点仪或跨步电压法进行精确定位"{tuple_delimiter}"子步骤"{tuple_delimiter}"地面精确定点的具体操作：在电缆路径上进行精确定位"){record_delimiter}
("entity"{tuple_delimiter}"协调资源进行开挖"{tuple_delimiter}"子步骤"{tuple_delimiter}"开挖与修复的具体操作：准备开挖工作"){record_delimiter}
("entity"{tuple_delimiter}"切除故障段"{tuple_delimiter}"子步骤"{tuple_delimiter}"开挖与修复的具体操作：使用工具切除损坏部分"){record_delimiter}
("entity"{tuple_delimiter}"严格执行电缆敷设规程避免弯曲半径过小和机械拖伤"{tuple_delimiter}"预防措施"{tuple_delimiter}"预防电缆主绝缘击穿故障的措施之一"){record_delimiter}
("entity"{tuple_delimiter}"低阻故障"{tuple_delimiter}"情况分支点"{tuple_delimiter}"根据击穿电阻高低分类的情况之一：电阻小于100欧姆"){record_delimiter}
("entity"{tuple_delimiter}"高阻故障"{tuple_delimiter}"情况分支点"{tuple_delimiter}"根据击穿电阻高低分类的情况之一：电阻可达兆欧级"){record_delimiter}
("relationship"{tuple_delimiter}"电缆"{tuple_delimiter}"主绝缘击穿故障"{tuple_delimiter}"故障分类"{tuple_delimiter}"主绝缘击穿是电缆的故障类型之一"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"主绝缘击穿故障"{tuple_delimiter}"绝缘电阻为零或极低"{tuple_delimiter}"故障现象"{tuple_delimiter}"主绝缘击穿故障表现为绝缘电阻为零或极低"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"主绝缘击穿故障"{tuple_delimiter}"绝缘老化"{tuple_delimiter}"原因为"{tuple_delimiter}"主绝缘击穿故障的原因可能是绝缘老化"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"主绝缘击穿故障"{tuple_delimiter}"低阻故障"{tuple_delimiter}"属于情况"{tuple_delimiter}"主绝缘击穿故障根据击穿电阻高低可分为低阻故障"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"主绝缘击穿故障"{tuple_delimiter}"高阻故障"{tuple_delimiter}"属于情况"{tuple_delimiter}"主绝缘击穿故障根据击穿电阻高低可分为高阻故障"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"地面精确定点"{tuple_delimiter}"声磁同步定点仪"{tuple_delimiter}"需要资源"{tuple_delimiter}"地面精确定点需要声磁同步定点仪"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"主绝缘击穿故障"{tuple_delimiter}"地面精确定点"{tuple_delimiter}"需执行"{tuple_delimiter}"抢修电缆主绝缘击穿故障需执行地面精确定点"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"主绝缘击穿故障"{tuple_delimiter}"开挖与修复"{tuple_delimiter}"需执行"{tuple_delimiter}"抢修电缆主绝缘击穿故障需执行开挖与修复"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"地面精确定点"{tuple_delimiter}"开挖与修复"{tuple_delimiter}"下一步"{tuple_delimiter}"地面精确定点完成后，下一步是开挖与修复"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"地面精确定点"{tuple_delimiter}"使用声磁同步定点仪或跨步电压法进行精确定位"{tuple_delimiter}"细节为"{tuple_delimiter}"地面精确定点的具体细节是使用声磁同步定点仪或跨步电压法进行精确定位"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"开挖与修复"{tuple_delimiter}"协调资源进行开挖"{tuple_delimiter}"细节为"{tuple_delimiter}"开挖与修复的具体细节是协调资源进行开挖"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"开挖与修复"{tuple_delimiter}"切除故障段"{tuple_delimiter}"细节为"{tuple_delimiter}"开挖与修复的具体细节是切除故障段"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"严格执行电缆敷设规程避免弯曲半径过小和机械拖伤"{tuple_delimiter}"绝缘老化"{tuple_delimiter}"可预防"{tuple_delimiter}"严格执行电缆敷设规程可以预防绝缘老化导致的故障"{tuple_delimiter}8){completion_delimiter}
#############################

！！！-必须执行的规则-
1. 禁止输出 JSON、Markdown、列表或任何其他特殊格式。
2. 必须使用格式：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
3. 必须使用格式：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
4. 实体描述不能为空，必须根据上下文总结实体的属性或行为；关系描述也不能为空，必须对关系进行说明
5.注意所有字符串字段都必须使用双引号包围
6.可能会出现某些实体类型和关系类型未出现，这种情况是正常的。
7.实体抽取时注意实体类型对应的约束，如“电气设备”的约束限制；关系抽取需要通过语义理解进行，尤其是未连接任意关系的实体，可能由于实体间距离较远难以发现，如<可预防>关系对应的首尾实体。

"""

human_template_build_graph = """
-真实数据- 
###################### 
实体类型：{entity_types}
关系类型：{relationship_types}
文本：{input_text} 
###################### 
1、请严格按照指定的元组格式输出，严禁使用 JSON、Markdown 块或其他格式。
2、必须使用格式：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
3、必须使用格式：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
4、实体描述不能为空，必须根据上下文总结实体的属性或行为；关系描述也不能为空，必须对关系进行说明。
5、注意所有字符串字段都必须使用双引号包围。
6、实体抽取时注意实体类型对应的约束，如“电气设备”的约束限制；关系抽取需要通过语义理解进行，尤其是未连接任意关系的实体，可能由于实体间距离较远难以发现，如<可预防>关系对应的首尾实体。
直接开始输出：
输出：
"""

system_template_build_index = """
你是一名数据处理助理。您的任务是识别列表中的重复实体，并决定应合并哪些实体。 
这些实体在格式或内容上可能略有不同，但本质上指的是同一个实体。运用你的分析技能来确定重复的实体。 
以下是识别重复实体的规则：
1.类型一致性原则（至关重要）：只有当两个实体属于同一个“实体类型”（即方括号内的类型相同）且所表达的是同一实体时，才能合并。
   - 例如："变压器1号 [设备]" 和 "1#变压器 [设备]" -> 可以合并。
   - 例如："变压器 [设备]" 和 "变压器故障 [故障表现]" -> **禁止合并**。
   - 例如："安全员 [抢修人员]" 和 "后勤人员 [抢修人员]" -> **禁止合并**。
2.语义上完全一致的实体应被视为重复。 
3.格式不同但内容相同的实体应被视为重复。 
4.引用同一现实世界对象或概念的实体，即使描述不同，也应被视为重复。 
5.如果它指的是不同的数字、日期或产品型号，请不要合并实体。
6.如果你无法 100% 确定两个实体指向同一个现实对象，严禁合并。
输出格式：
1.将要合并的实体输出为Python列表的格式，输出时保持它们输入时的原文。
2.如果有多组可以合并的实体，每组输出为一个单独的列表，每组分开输出为一行。
3.如果没有要合并的实体，就输出一个空的列表。
4.只输出列表即可，不需要其它的说明。
5.不要输出嵌套的列表，只输出列表。
###################### 
-示例- 
###################### 
Example 1:
Input: ['检修班[抢修人员]', '高试班 [抢修人员]', '检修班成员 [抢修人员]']
Output:
['检修班', '检修班成员']

Example 2 (类型冲突示例):
Input: ['变压器 [电气设备]', '变压器油温过高 [故障表现]']
Output:
[]

Example 3:
Input: ['断路器 [电气设备]', '10kV断路器 [电气设备]', '变压器 [电气设备]']
Output:
['断路器', '10kV断路器']
#############################
"""

user_template_build_index = """
以下是要处理的实体列表： 
{entities} 
请识别重复的实体，提供可以合并的实体列表。
输出：
"""

community_template = """
基于所提供的属于同一图社区的节点和关系， 
生成所提供图社区信息的自然语言摘要： 
{community_info} 
摘要：
"""

COMMUNITY_SUMMARY_PROMPT = """
你是一个知识图谱摘要生成专家。请根据提供的图社区（节点和关系）信息，生成一段简洁的中文摘要。
要求：
1. 不超过 100 字，尽量简洁描述社区核心特征。
2. 进行适当的归纳总结。
3.描述到社区的节点与关系时，请保留其全称不要缩写。
"""

entity_alignment_prompt = """
Given these entities that should refer to the same concept:
{entity_desc}

Which entity ID best represents the canonical form? Reply with only the entity ID."""

__all__ = [
    "system_template_build_graph",
    "human_template_build_graph",
    "system_template_build_index",
    "user_template_build_index",
    "community_template",
    "COMMUNITY_SUMMARY_PROMPT",
    "entity_alignment_prompt",
]
