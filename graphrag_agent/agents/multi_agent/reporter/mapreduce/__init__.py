import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

"""
Map-Reduce reporter组件集合。

该子模块提供证据映射、章节归约与报告组装的能力，用于支持长文档写作。
"""

from .evidence_mapper import EvidenceMapper, EvidenceSummary
from .section_reducer import SectionReducer, ReduceStrategy
from .report_assembler import ReportAssembler

__all__ = [
    "EvidenceMapper",
    "EvidenceSummary",
    "SectionReducer",
    "ReduceStrategy",
    "ReportAssembler",
]
