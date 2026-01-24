import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

"""
Reporter层对外接口
"""

from graphrag_agent.agents.multi_agent.reporter.base_reporter import (
    BaseReporter,
    ReporterConfig,
    ReportResult,
    SectionContent,
)
from graphrag_agent.agents.multi_agent.reporter.outline_builder import (
    OutlineBuilder,
    ReportOutline,
    SectionOutline,
)
from graphrag_agent.agents.multi_agent.reporter.section_writer import (
    SectionWriter,
    SectionWriterConfig,
    SectionDraft,
)
from graphrag_agent.agents.multi_agent.reporter.consistency_checker import (
    ConsistencyChecker,
    ConsistencyCheckResult,
)
from graphrag_agent.agents.multi_agent.reporter.formatter import CitationFormatter

__all__ = [
    "BaseReporter",
    "ReporterConfig",
    "ReportResult",
    "SectionContent",
    "OutlineBuilder",
    "ReportOutline",
    "SectionOutline",
    "SectionWriter",
    "SectionWriterConfig",
    "SectionDraft",
    "ConsistencyChecker",
    "ConsistencyCheckResult",
    "CitationFormatter",
]
