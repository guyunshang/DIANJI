import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 统一加载环境变量，确保配置来源一致
load_dotenv()


def _get_env_int(key: str, default: Optional[int]) -> Optional[int]:
    """获取整型环境变量，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 需要整数值，但当前为 {raw}") from exc


def _get_env_float(key: str, default: Optional[float]) -> Optional[float]:
    """获取浮点型环境变量，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 需要浮点值，但当前为 {raw}") from exc


def _get_env_bool(key: str, default: bool) -> bool:
    """获取布尔型环境变量，支持 true/false/1/0 表达式"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _get_env_choice(key: str, choices: set[str], default: str) -> str:
    """获取有限集合中的字符串配置，未设置时返回默认值"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value not in choices:
        raise ValueError(
            f"环境变量 {key} 必须为 {', '.join(sorted(choices))} 之一，但当前为 {raw}"
        )
    return value


# ===== 基础路径设置 =====

BASE_DIR = Path(__file__).resolve().parent.parent  # graphrag_agent包目录
PROJECT_ROOT = BASE_DIR.parent  # 项目根目录
FILES_DIR = PROJECT_ROOT / "files"
FILE_REGISTRY_PATH = PROJECT_ROOT / "file_registry.json"  # 文件注册表路径

# ===== 知识库与系统参数 =====

KB_NAME = "电气设备故障及处理"  # 知识库主题，用于deepsearch
workers = _get_env_int("FASTAPI_WORKERS", 2) or 2  # FastAPI 并发进程数

# ===== 知识图谱配置 =====

theme = "电气设备故障及处理知识图谱"  # 知识图谱主题

entity_types = [
    "电气设备",
    "故障类型",
    "故障表现",
    "故障原因",
    "抢修人员",
    "抢修资源",
    "抢修方法",
    "子步骤",
    "预防措施",
    "情况分支点"
]  # 知识图谱实体类型

entity_definitions = {
    "电气设备": "只包括以下设备：变压器、电缆、杆塔、输电线路、避雷器、光缆、环网柜、断路器、断路器，其他不是上述之一的一概禁止抽取为此类型。",
    "故障类型": "电气设备可能发生的故障的类型或名称，如匝间短路故障。",
    "故障表现": "某种故障类型对应的具体异常状态或现象，如变压器匝间短路时对应的油温异常升高、乙炔含量增长等。",
    "故障原因": "导致电气设备某种故障类型或故障表现发生的直接或间接诱因。如变压器的短路匝环流是导致油温异常升高的故障原因",
    "抢修人员": "可以是抢修团队、团队中的个人，也可以是其他参与抢修的人员。注意指的是比较明确的对象，如某班组、某人此类具体的应该抽取；如试验人员、运行人员、操作人员这些比较宽泛的禁止抽取",
    "抢修资源": "抢修过程中涉及到的工具、材料、仪器仪表或安全用具，如吊车、真空泵等。注意“吊车、真空泵”应视为两个抢修资源，不要因为连接符号而视为一个抢修资源。另外人员、方法（如低压脉冲法）等都不是资源。",
    "抢修方法": "针对某种故障抢修要采取的处理步骤、维修手段或操作流程。注意指的是抢修流程或方法，抢修资源禁止归为此类。",
    "子步骤": "抢修方法对应的具体操作细节或流程",
    "预防措施": "为了防止故障再次发生或减少故障概率而采取的管理或技术措施。",
    "情况分支点": "需要分情况或者分类时的点，如<电缆主绝缘击穿故障可分为高阻故障与低阻故障>，则高阻故障与低阻故障为情况分支点。如<对于金属护套破损，使用电缆护层故障测试仪定位破损点，对于疑似进水的电缆，剖开外护套检查潮气痕迹>，则金属护套破损与疑似进水为情况分支点。如<母线按使用材料分，可分为铜母线、铝母线和钢母线>，则铜母线、铝母线和钢母线为情况分支点"
}

relationship_types = [
    "故障分类",
    "表现为",
    "原因为",
    "分配维修",
    "需要资源",
    "需执行",
    "负责",
    "下一步",
    "细节为",
    "可预防",
    "属于情况"
]  # 知识图谱关系类型

# 关系定义：解释关系的含义，并限制其连接的实体类型（源 -> 目标）
relationship_definitions = {
    "故障分类": "描述电气设备存在或出现了某种故障类型。 (源: 电气设备 -> 目标: 故障类型)",
    "表现为": "描述电气设备的某种故障对应的具体故障现象或特征。优先描述故障类型对应故障表现，若无故障类型，则描述电气设备对应故障表现。 (源: 故障类型/电气设备 -> 目标: 故障表现)",
    "原因为": "描述导致电气设备某种故障类型或故障表现的具体原因。 (源: 故障类型/故障表现 -> 目标: 故障原因)",
    "分配维修": "描述团队或人员被指派负责维修某设备或处理某故障。 (源: 抢修人员 -> 目标: 故障类型)",
    "需要资源": "描述抢修某故障所需要的物资和工具。 (源: 抢修方法 -> 目标:抢修资源)",
    "需执行": "描述解决特定的故障类型或故障表现需要执行某些方法或措施，步骤较多则分步骤书写。优先描述故障类型对应抢修方法，若无故障类型，则描述故障表现对应抢修方法。 (源: 故障类型/故障表现 -> 目标: 抢修方法)",
    "负责": "描述某些或某个抢修人员所负责的/执行的抢修方法。 (源: 抢修人员 -> 目标: 抢修方法)",
    "下一步": "在抢修方法中有明显的逻辑次序且步骤较多则进行分步骤描述，表示抢修方法的次序关系。 (源: 抢修方法 -> 源: 抢修方法)",
    "细节为": "描述抢修方法的具体操作，如快速隔离与初步测距的具体细节为隔离故障电缆两端并验电接地、脉冲法进行波形分析。 (源: 抢修方法 -> 源: 子步骤)",
    "可预防": "描述措施可以预防某种故障。 (源: 预防措施 -> 目标: 故障类型/故障表现/故障原因)",
    "属于情况":"描述分类或者分情况的情形。如<母线按使用材料分，可分为铜母线、铝母线和钢母线>中，母线有铜母线、铝母线和钢母线3种情况。 (源: 故障类型/故障表现/抢修方法 -> 目标:情况分支点)"
}

# 冲突解决策略：manual_first / auto_first / merge
conflict_strategy = os.getenv("GRAPH_CONFLICT_STRATEGY", "manual_first")

# 社区检测算法：leiden / sllpa
community_algorithm = os.getenv("GRAPH_COMMUNITY_ALGORITHM", "leiden")

# ===== 文本处理配置 =====

CHUNK_SIZE = _get_env_int("CHUNK_SIZE", 400) or 400  # 文本分块大小
OVERLAP = _get_env_int("CHUNK_OVERLAP", 0) or 0  # 分块重叠长度
MAX_TEXT_LENGTH = _get_env_int("MAX_TEXT_LENGTH", 500000) or 500000  # 最大文本长度
similarity_threshold = _get_env_float("SIMILARITY_THRESHOLD", 0.85) or 0.85  # 向量相似度阈值

# ===== 回答生成配置 =====

response_type = os.getenv("RESPONSE_TYPE", "多个段落")  # 默认回答形式

# ===== Agent 工具描述 =====

lc_description = (
    "用于需要具体细节的查询。检索电气设备故障的具体信息以及故障原因和所需资源等的具体流程。"
    "适用于'某个具体故障是什么原因导致的'、'抢修某种故障需要什么资源'等问题。"
    "模型【严禁直接回答】，必须调用 local_search_structured 以获取图谱中的专业参考数据。没有此数据回答将不可靠。"
)
gl_description = (
    "用于需要总结归纳的查询。分析电气设备故障的整体框架、处理流程、原因总结等宏观内容。"
    "适用于'故障的处理流程'、'故障的原因可能是'等需要系统性分析的问题。适用于'故障的处理流程'、'故障的原因可能是'等需要系统性分析的问题。"
)
naive_description = (
    "基础检索工具，直接查找与问题最相关的文本片段，不做复杂分析。快速获取关于故障信息、故障原因等的信息，返回最匹配的原文段落。"
)

examples = [
    "变压器有哪些常见故障？",
    "如何预防变压器油位异常？",
    "断路器拒动需要准备哪些资源？",
    "主变冷却器故障如何处置？"
]  # 前端示例问题

# ===== 性能优化配置 =====

MAX_WORKERS = _get_env_int("MAX_WORKERS", 4) or 4  # 并行工作线程数
BATCH_SIZE = _get_env_int("BATCH_SIZE", 100) or 100  # 批处理大小
ENTITY_BATCH_SIZE = _get_env_int("ENTITY_BATCH_SIZE", 50) or 50  # 实体批次大小
CHUNK_BATCH_SIZE = _get_env_int("CHUNK_BATCH_SIZE", 100) or 100  # 文本块批次
EMBEDDING_BATCH_SIZE = _get_env_int("EMBEDDING_BATCH_SIZE", 64) or 64  # 向量批次
LLM_BATCH_SIZE = _get_env_int("LLM_BATCH_SIZE", 5) or 5  # LLM 批次
COMMUNITY_BATCH_SIZE = _get_env_int("COMMUNITY_BATCH_SIZE", 50) or 50  # 社区批次大小

# ===== GDS 相关配置 =====

GDS_MEMORY_LIMIT = _get_env_int("GDS_MEMORY_LIMIT", 6) or 6  # GDS 内存限制(GB)
GDS_CONCURRENCY = _get_env_int("GDS_CONCURRENCY", 4) or 4  # GDS 并发度
GDS_NODE_COUNT_LIMIT = _get_env_int("GDS_NODE_COUNT_LIMIT", 50000) or 50000  # 节点上限
GDS_TIMEOUT_SECONDS = _get_env_int("GDS_TIMEOUT_SECONDS", 300) or 300  # 超时时长

# ===== 实体消歧与对齐配置 =====

DISAMBIG_STRING_THRESHOLD = _get_env_float("DISAMBIG_STRING_THRESHOLD", 0.7) or 0.7
DISAMBIG_VECTOR_THRESHOLD = _get_env_float("DISAMBIG_VECTOR_THRESHOLD", 0.85) or 0.85
DISAMBIG_NIL_THRESHOLD = _get_env_float("DISAMBIG_NIL_THRESHOLD", 0.6) or 0.6
DISAMBIG_TOP_K = _get_env_int("DISAMBIG_TOP_K", 5) or 5

ALIGNMENT_CONFLICT_THRESHOLD = (
    _get_env_float("ALIGNMENT_CONFLICT_THRESHOLD", 0.8) or 0.8
)
ALIGNMENT_MIN_GROUP_SIZE = _get_env_int("ALIGNMENT_MIN_GROUP_SIZE", 2) or 2

# ===== 路径与缓存配置 =====

DEFAULT_CACHE_ROOT = Path(
    os.getenv("CACHE_ROOT", PROJECT_ROOT / "cache")
).expanduser()
MODEL_CACHE_ROOT = Path(
    os.getenv("MODEL_CACHE_ROOT", DEFAULT_CACHE_ROOT)
).expanduser()
MODEL_CACHE_DIR = MODEL_CACHE_ROOT / "model"
CACHE_DIR = Path(os.getenv("CACHE_DIR", DEFAULT_CACHE_ROOT)).expanduser()
TIKTOKEN_CACHE_DIR = Path(
    os.getenv("TIKTOKEN_CACHE_DIR", DEFAULT_CACHE_ROOT / "tiktoken")
).expanduser()
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))

SENTENCE_TRANSFORMER_MODELS = [
    item.strip()
    for item in os.getenv("SENTENCE_TRANSFORMER_MODELS", "").split(",")
    if item.strip()
]  # 预加载的本地模型列表
CACHE_EMBEDDING_PROVIDER = os.getenv(
    "CACHE_EMBEDDING_PROVIDER", "sentence_transformer"
).lower()
CACHE_SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "CACHE_SENTENCE_TRANSFORMER_MODEL", "D:/bge-m3/BAAI/bge-m3"
)

CACHE_SETTINGS = {
    "dir": CACHE_DIR,
    "memory_only": _get_env_bool("CACHE_MEMORY_ONLY", True),
    "max_memory_size": _get_env_int("CACHE_MAX_MEMORY_SIZE", 100) or 100,
    "max_disk_size": _get_env_int("CACHE_MAX_DISK_SIZE", 1000) or 1000,
    "thread_safe": _get_env_bool("CACHE_THREAD_SAFE", True),
    "enable_vector_similarity": _get_env_bool(
        "CACHE_ENABLE_VECTOR_SIMILARITY", True
    ),
    "similarity_threshold": _get_env_float(
        "CACHE_SIMILARITY_THRESHOLD", similarity_threshold
    )
    or similarity_threshold,
    "max_vectors": _get_env_int("CACHE_MAX_VECTORS", 10000) or 10000,
}

# ===== Neo4j 连接配置 =====

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_MAX_POOL_SIZE = _get_env_int("NEO4J_MAX_POOL_SIZE", 10) or 10
NEO4J_REFRESH_SCHEMA = _get_env_bool("NEO4J_REFRESH_SCHEMA", False)

NEO4J_CONFIG = {
    "uri": NEO4J_URI,
    "username": NEO4J_USERNAME,
    "password": NEO4J_PASSWORD,
    "max_pool_size": NEO4J_MAX_POOL_SIZE,
    "refresh_schema": NEO4J_REFRESH_SCHEMA,
}

# ===== LLM 与嵌入模型配置 =====

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL") or None
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL") or None
LLM_TEMPERATURE = _get_env_float("TEMPERATURE", None)
LLM_MAX_TOKENS = _get_env_int("MAX_TOKENS", None)

OPENAI_EMBEDDING_CONFIG = {
    "model": OPENAI_EMBEDDINGS_MODEL,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
}

OPENAI_LLM_CONFIG = {
    "model": OPENAI_LLM_MODEL,
    "temperature": LLM_TEMPERATURE,
    "max_tokens": LLM_MAX_TOKENS,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
}

# ===== 相似实体检测参数 =====

SIMILAR_ENTITY_SETTINGS = {
    "word_edit_distance": _get_env_int("SIMILAR_ENTITY_WORD_EDIT_DISTANCE", 1) or 1,
    "batch_size": _get_env_int("SIMILAR_ENTITY_BATCH_SIZE", 700) or 700,
    "memory_limit": _get_env_int(
        "SIMILAR_ENTITY_MEMORY_LIMIT", GDS_MEMORY_LIMIT
    )
    or GDS_MEMORY_LIMIT,
    "top_k": _get_env_int("SIMILAR_ENTITY_TOP_K", 10) or 10,
}

# ===== 搜索工具配置 =====

BASE_SEARCH_CONFIG = {
    "cache_max_size": _get_env_int("SEARCH_CACHE_MEMORY_SIZE", 200) or 200,
    "vector_limit": _get_env_int("SEARCH_VECTOR_LIMIT", 5) or 5,
    "text_limit": _get_env_int("SEARCH_TEXT_LIMIT", 5) or 5,
    "semantic_top_k": _get_env_int("SEARCH_SEMANTIC_TOP_K", 5) or 5,
    "relevance_top_k": _get_env_int("SEARCH_RELEVANCE_TOP_K", 5) or 5,
}

LOCAL_SEARCH_SETTINGS = {
    "top_chunks": _get_env_int("LOCAL_SEARCH_TOP_CHUNKS", 3) or 3,
    "top_communities": _get_env_int("LOCAL_SEARCH_TOP_COMMUNITIES", 3) or 3,
    "top_outside_relationships": _get_env_int(
        "LOCAL_SEARCH_TOP_OUTSIDE_RELS", 10
    )
    or 10,
    "top_inside_relationships": _get_env_int(
        "LOCAL_SEARCH_TOP_INSIDE_RELS", 10
    )
    or 10,
    "top_entities": _get_env_int("LOCAL_SEARCH_TOP_ENTITIES", 10) or 10,
    "index_name": os.getenv("LOCAL_SEARCH_INDEX_NAME", "vector"),
}

GLOBAL_SEARCH_SETTINGS = {
    "default_level": _get_env_int("GLOBAL_SEARCH_LEVEL", 0) or 0,
    "community_batch_size": _get_env_int("GLOBAL_SEARCH_BATCH_SIZE", 5) or 5,
}

NAIVE_SEARCH_TOP_K = _get_env_int("NAIVE_SEARCH_TOP_K", 3) or 3

HYBRID_SEARCH_SETTINGS = {
    "entity_limit": _get_env_int("HYBRID_SEARCH_ENTITY_LIMIT", 15) or 15,
    "max_hop_distance": _get_env_int("HYBRID_SEARCH_MAX_HOP", 2) or 2,
    "top_communities": _get_env_int("HYBRID_SEARCH_TOP_COMMUNITIES", 3) or 3,
    "batch_size": _get_env_int("HYBRID_SEARCH_BATCH_SIZE", 10) or 10,
    "community_level": _get_env_int("HYBRID_SEARCH_COMMUNITY_LEVEL", 0) or 0,
}

# ===== Agent 配置 =====

AGENT_SETTINGS = {
    "default_recursion_limit": _get_env_int("AGENT_RECURSION_LIMIT", 5) or 5,
    "chunk_size": _get_env_int("AGENT_CHUNK_SIZE", 4) or 4,
    "stream_flush_threshold": _get_env_int("AGENT_STREAM_FLUSH_THRESHOLD", 40)
    or 40,
    "deep_stream_flush_threshold": _get_env_int(
        "DEEP_AGENT_STREAM_FLUSH_THRESHOLD", 80
    )
    or 80,
    "fusion_stream_flush_threshold": _get_env_int(
        "FUSION_AGENT_STREAM_FLUSH_THRESHOLD", 60
    )
    or 60,
}

# ===== 多智能体（Plan-Execute-Report）配置 =====

MULTI_AGENT_PLANNER_MAX_TASKS = _get_env_int("MA_PLANNER_MAX_TASKS", 6) or 6
MULTI_AGENT_ALLOW_UNCLARIFIED_PLAN = _get_env_bool("MA_ALLOW_UNCLARIFIED_PLAN", True)
MULTI_AGENT_DEFAULT_DOMAIN = os.getenv("MA_DEFAULT_DOMAIN", "通用")

MULTI_AGENT_AUTO_GENERATE_REPORT = _get_env_bool("MA_AUTO_GENERATE_REPORT", True)
MULTI_AGENT_STOP_ON_CLARIFICATION = _get_env_bool("MA_STOP_ON_CLARIFICATION", True)
MULTI_AGENT_STRICT_PLAN_SIGNAL = _get_env_bool("MA_STRICT_PLAN_SIGNAL", True)

MULTI_AGENT_DEFAULT_REPORT_TYPE = os.getenv("MA_DEFAULT_REPORT_TYPE", "long_document")
MULTI_AGENT_ENABLE_CONSISTENCY_CHECK = _get_env_bool(
    "MA_ENABLE_CONSISTENCY_CHECK", True
)
MULTI_AGENT_ENABLE_MAPREDUCE = _get_env_bool("MA_ENABLE_MAPREDUCE", True)
MULTI_AGENT_MAPREDUCE_THRESHOLD = _get_env_int("MA_MAPREDUCE_THRESHOLD", 20) or 20
MULTI_AGENT_MAX_TOKENS_PER_REDUCE = (
    _get_env_int("MA_MAX_TOKENS_PER_REDUCE", 4000) or 4000
)
MULTI_AGENT_ENABLE_PARALLEL_MAP = _get_env_bool("MA_ENABLE_PARALLEL_MAP", True)

MULTI_AGENT_SECTION_MAX_EVIDENCE = (
    _get_env_int("MA_SECTION_MAX_EVIDENCE", 8) or 8
)
MULTI_AGENT_SECTION_MAX_CONTEXT_CHARS = (
    _get_env_int("MA_SECTION_MAX_CONTEXT_CHARS", 800) or 800
)
MULTI_AGENT_REFLECTION_ALLOW_RETRY = _get_env_bool(
    "MA_REFLECTION_ALLOW_RETRY", False
)
MULTI_AGENT_REFLECTION_MAX_RETRIES = (
    _get_env_int("MA_REFLECTION_MAX_RETRIES", 1) or 1
)
MULTI_AGENT_WORKER_EXECUTION_MODE = _get_env_choice(
    "MA_WORKER_EXECUTION_MODE",
    {"sequential", "parallel"},
    "sequential",
)
MULTI_AGENT_WORKER_MAX_CONCURRENCY = (
    _get_env_int("MA_WORKER_MAX_CONCURRENCY", MAX_WORKERS) or MAX_WORKERS
)
if MULTI_AGENT_WORKER_MAX_CONCURRENCY < 1:
    raise ValueError("MA_WORKER_MAX_CONCURRENCY 必须大于等于 1")
