import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

import re
from typing import List, Optional

# 直接调用 settings.py 中的原始定义
from graphrag_agent.config.settings import (
    entity_definitions,
    relationship_types
)
from graphrag_agent.config.prompts.search_prompts import ENTITY_EXTRACTION_WITH_SCHEMA_PROMPT

def extract_between(text: str, start_marker: str, end_marker: str) -> List[str]:
    """
    提取起始和结束标记之间的内容
    
    参数:
        text: 要搜索的文本
        start_marker: 起始标记
        end_marker: 结束标记
        
    返回:
        List[str]: 提取的内容字符串列表
    """
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
    return re.findall(pattern, text, flags=re.DOTALL)

def extract_from_templates(text: str, templates: List[str], regex: bool = False) -> List[str]:
    """
    基于带占位符的模板提取内容
    
    参数:
        text: 要搜索的文本
        templates: 带{}占位符的模板字符串列表
        regex: 是否将模板作为正则表达式处理
        
    返回:
        List[str]: 提取的内容字符串列表
    """
    results = []
    
    for template in templates:
        if regex:
            # 直接使用模板作为正则表达式
            matches = re.findall(template, text, re.DOTALL)
            results.extend(matches)
        else:
            # 将模板转换为正则表达式（通过转义和替换占位符）
            pattern = template.replace("{}", "(.*?)")
            pattern = re.escape(pattern).replace("\\(\\*\\*\\?\\)", "(.*?)")
            matches = re.findall(pattern, text, re.DOTALL)
            results.extend(matches)
    
    return results

def extract_sentences(text: str, max_sentences: Optional[int] = None) -> List[str]:
    """
    从文本中提取句子
    
    参数:
        text: 要提取句子的文本
        max_sentences: 最大提取句子数
        
    返回:
        List[str]: 句子列表
    """
    if not text:
        return []
    
    # 简单的句子分割（可以使用NLP库进行改进）
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    
    # 移除空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if max_sentences:
        return sentences[:max_sentences]
    return sentences


class NLPProcessor:
    def __init__(self, llm):
        self.llm = llm
        # 直接将字典和列表转为字符串，供 Prompt 调用
        self.entity_schema = str(entity_definitions)
        self.relation_schema = str(relationship_types)

    def extract_keywords(self, query: str) -> List[str]:
        """基于本体感知提取关键词"""
        # 填充 Prompt
        prompt = ENTITY_EXTRACTION_WITH_SCHEMA_PROMPT.format(
            entity_definitions=self.entity_schema,
            relationship_types=self.relation_schema,
            query=query
        )

        # 调用大模型并清洗结果
        response = self.llm.invoke(prompt)
        content = response.content.strip()

        # 处理中英文逗号并转为列表
        content = content.replace("，", ",")
        if "," in content:
            return [k.strip() for k in content.split(",") if k.strip()]
        return [content] if content else []