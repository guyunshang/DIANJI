import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import re
from typing import List
import streamlit as st

def extract_source_ids(answer: str) -> List[str]:
    """从回答中提取引用的源ID"""
    source_ids = []

    if not answer:
        return []

    # 1. 优先尝试提取 HTML 注释中的标准 JSON
    html_json_pattern = r''
    html_matches = re.findall(html_json_pattern, answer, re.DOTALL)

    for json_str in html_matches:
        try:
            data = json.loads(json_str)
            # 路径 data -> Chunks 或 data -> data -> Chunks
            if "data" in data and isinstance(data["data"], dict):
                chunks = data["data"].get("Chunks", [])
            else:
                chunks = data.get("Chunks", [])

            if isinstance(chunks, list):
                source_ids.extend([str(c) for c in chunks])
        except:
            pass

    chunks_pattern = r"['\"]?Chunks['\"]?\s*:\s*\[([^\]]*)\]"
    matches = re.findall(chunks_pattern, answer)

    if matches:
        for match in matches:
            # 提取双引号包围的 ID "id"
            double_quoted = re.findall(r'"([^"]*)"', match)
            if double_quoted:
                source_ids.extend(double_quoted)

            # 提取单引号包围的 ID 'id'
            single_quoted = re.findall(r"'([^']*)'", match)
            if single_quoted:
                source_ids.extend(single_quoted)

            # 如果没有引号，尝试逗号分割 (针对数字ID)
            if not double_quoted and not single_quoted:
                ids = [id.strip() for id in match.split(',') if id.strip()]
                source_ids.extend(ids)

    # 3. 去重并过滤空值
    unique_ids = list(set([sid for sid in source_ids if sid]))
    return unique_ids

def display_source_content(content: str):
    """更好地显示源内容"""
    st.markdown("""
    <style>
    .source-content {
        white-space: pre-wrap;
        overflow-x: auto;
        font-family: monospace;
        line-height: 1.6;
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 15px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e1e4e8;
        color: #24292e;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 将换行符转换为HTML换行，确保格式正确
    formatted_content = content.replace("\n", "<br>")
    st.markdown(f'<div class="source-content">{formatted_content}</div>', unsafe_allow_html=True)


def process_thinking_content(content: str, show_thinking: bool = False):
    """
    处理带有思考过程的内容
    
    Args:
        content: 原始内容
        show_thinking: 是否显示思考过程
        
    Returns:
        dict: 包含处理后的内容信息
    """
    if not isinstance(content, str):
        return {"processed": content, "has_thinking": False}
        
    # 检查是否有思考过程
    if "<think>" in content and "</think>" in content:
        # 使用正则表达式提取思考过程
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            thinking_process = think_match.group(1).strip()
            # 移除思考过程部分，只保留答案
            answer_only = content.replace(f"<think>{thinking_process}</think>", "").strip()
            
            # 将思考过程格式化为Markdown引用格式
            thinking_lines = thinking_process.split('\n')
            quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
            
            return {
                "processed": answer_only,
                "has_thinking": True,
                "thinking": quoted_thinking,
                "original": content
            }
    
    # 如果没有思考过程或提取失败，返回原内容
    return {"processed": content, "has_thinking": False}