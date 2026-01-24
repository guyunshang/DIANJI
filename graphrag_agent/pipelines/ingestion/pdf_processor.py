import sys
from pathlib import Path

# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

import fitz
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from typing import Optional


def pdf_to_html(input_path: str) -> str:
    """
    将PDF转换为HTML字符串，保留原始布局结构
    """
    doc = fitz.open(input_path)

    # 创建基本的HTML结构
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>PDF Extract</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .page { margin-bottom: 20px; }
            .paragraph { margin-bottom: 12px; }
        </style>
    </head>
    <body>
    '''

    current_paragraph = ""
    prev_block_bottom = None
    prev_block_size = None
    prev_block_bbox = None
    paragraph_started = False
    current_sep_type = "double"  # 默认为双换行

    for page_num, page in enumerate(tqdm(doc, desc="转换进度")):
        blocks = page.get_text("dict")["blocks"]
        html_content += f'<div class="page" id="page-{page_num + 1}">\n'

        for block in blocks:
            if block["type"] == 0:  # 仅处理文本块
                # 提取当前块字号
                current_size = block["lines"][0]["spans"][0]["size"] if block["lines"] and block["lines"][0][
                    "spans"] else 12

                # 提取文本
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        font = span["font"]
                        size = span["size"]
                        color = span["color"]
                        block_text += f'<span style="font-family: {font}; font-size: {size}pt; color: #{color:06x};">{text}</span>'
                    block_text += " "

                # 段落判定逻辑
                is_new_paragraph = False
                temp_sep = "double"  # 判定过程中临时的间隔类型

                if prev_block_bottom is not None:
                    # 判定1：纵向间距判定（优先级最高）
                    # 注意：跨页时 gap 会是负数，会自动判定为非新段落，从而实现跨页合并
                    gap = block["bbox"][1] - prev_block_bottom
                    v_threshold = current_size * 1.0
                    if prev_block_size:
                        v_threshold = min(v_threshold, prev_block_size)

                    if gap > v_threshold:
                        is_new_paragraph = True
                        temp_sep = "double"

                    # 判定2：缩进判定（仅在间距判定不成立时）
                    if not is_new_paragraph and prev_block_bbox:
                        # 确保已经换行（容差2像素）
                        is_new_line = block["bbox"][1] >= prev_block_bottom - 2
                        if is_new_line:
                            # 判定缩进：当前块左边界明显右移
                            if block["bbox"][0] > prev_block_bbox[0] + current_size * 1.5:
                                is_new_paragraph = True
                                temp_sep = "single"

                # 执行分段
                if is_new_paragraph or not paragraph_started:
                    if current_paragraph:
                        # 写入上一段及它对应的间隔标记
                        html_content += f'<div class="paragraph" data-sep="{current_sep_type}">{current_paragraph}</div>\n'
                    current_paragraph = block_text
                    paragraph_started = True
                    current_sep_type = temp_sep  # 为下一段准备间隔类型
                else:
                    current_paragraph += block_text

                # 更新状态位（必须在 block 循环内更新）
                prev_block_bottom = block["bbox"][3]
                prev_block_size = current_size
                prev_block_bbox = block["bbox"]

        html_content += '</div>\n'  # 页面结束

    # 处理全文最后一个段落
    if current_paragraph:
        html_content += f'<div class="paragraph" data-sep="{current_sep_type}">{current_paragraph}</div>\n'

    html_content += "</body></html>"
    doc.close()
    return html_content


def html_to_text(html_content: str) -> str:
    """
    基于 data-sep 属性动态生成单/双换行符
    """
    soup = BeautifulSoup(html_content, "html.parser")
    paragraphs = soup.find_all('div', class_='paragraph')

    text_elements = []
    for i, para in enumerate(paragraphs):
        content = para.get_text()
        content = re.sub(r'\s+', ' ', content).strip()
        if not content:
            continue

        sep = para.get('data-sep', 'double')

        if i == 0:
            text_elements.append(content)
        else:
            # 缩进段落用 \n，间距段落用 \n\n
            joiner = "\n" if sep == "single" else "\n\n"
            text_elements.append(joiner + content)

    return "".join(text_elements)


def pdf_to_text(input_path: str) -> str:
    html_content = pdf_to_html(input_path)
    text = html_to_text(html_content)
    return text