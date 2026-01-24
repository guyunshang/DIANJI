import sys
import os
import re
from pathlib import Path

# 1. 确保项目根目录在系统路径中
# 假设脚本运行在 F:\Pycharm\PycharmProjects\TAO\graph-rag-agent-new\
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# 导入项目中现有的组件
try:
    from graphrag_agent.pipelines.ingestion.file_reader import FileReader
    from graphrag_agent.pipelines.ingestion.text_chunker import ChineseTextChunker
except ImportError:
    print("导入失败，请确保在项目根目录下运行此脚本。")
    sys.exit(1)


def tokens_to_text(tokens) -> str:
    """将 tokens 列表拼回字符串（逻辑同 text_chunker 内的方法）"""
    if not tokens:
        return ""
    ascii_pattern = re.compile(r'^[a-zA-Z0-9]+$')
    result = []
    prev_is_ascii = False
    for token in tokens:
        current_is_ascii = bool(ascii_pattern.match(token))
        if prev_is_ascii and current_is_ascii:
            result.append(' ')
        result.append(token)
        prev_is_ascii = current_is_ascii
    return ''.join(result)


def run_chunk_test():
    # ====== 配置路径 ======
    # 输入文件路径
    input_file_path = r"F:\Pycharm\PycharmProjects\TAO\graph-rag-agent-new\files\电气设备故障及处理方法.docx"
    # 输出桌面路径
    desktop_dir = Path(r"C:\Users\HNQ\Desktop")
    output_folder = desktop_dir / "分块测试结果"
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"--- 开始测试分块逻辑 ---")
    print(f"读取文件: {input_file_path}")

    # 2. 读取文件内容
    # 获取目录路径和文件名
    dir_path = str(Path(input_file_path).parent)
    file_name = os.path.basename(input_file_path)
    ext = os.path.splitext(file_name)[1].lower()  # 获取文件后缀

    reader = FileReader(dir_path)

    print(f"检测到文件类型: {ext}")

    # 根据后缀动态选择读取方法
    if ext == '.pdf':
        full_text = reader._read_pdf(input_file_path)
    elif ext == '.docx':
        full_text = reader._read_docx(input_file_path)
    elif ext == '.doc':
        full_text = reader._read_doc(input_file_path)
    elif ext in ['.txt', '.md']:
        full_text = reader._read_txt(input_file_path)
    else:
        print(f"错误: 不支持的文件类型 {ext}")
        return

    if not full_text or "无法读取" in full_text:
        print(f"错误: 无法读取文件内容。")
        return

    print(f"文件读取成功，总长度: {len(full_text)} 字符")

    # 3. 初始化分块器
    # 注意：这里传入你要求的 700 和 100（overlap 设为 100）
    # 确保你的 text_chunker.py 已经按照之前的建议修改了 _find_chunk_boundary
    chunker = ChineseTextChunker(
        chunk_size=400,
        overlap=100
    )

    # 4. 执行分块
    print("正在进行分块处理...")
    chunks_tokens = chunker.chunk_text(full_text)

    if not chunks_tokens:
        print("未生成任何分块，请检查分块器逻辑。")
        return

    print(f"分块完成，共生成 {len(chunks_tokens)} 个文本块")

    # 5. 输出结果到桌面
    print(f"保存分块文件至: {output_folder}")
    for idx, tok_chunk in enumerate(chunks_tokens, start=1):
        text_content = tokens_to_text(tok_chunk)

        # 检查该块结尾是否符合优先级
        end_mark = "未知"
        if text_content.endswith('\n\n'):
            end_mark = "\\n\\n (段落)"
        elif text_content.endswith('\n'):
            end_mark = "\\n (换行)"
        elif any(text_content.endswith(p) for p in ['。', '！', '？', '；']):
            end_mark = "标点符号"
        else:
            end_mark = "强制截断"

        file_name = output_folder / f"chunk_{idx:03d}.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"--- Chunk {idx} | 长度: {len(text_content)} | 结尾匹配: {end_mark} ---\n")
            f.write(text_content)

    print(f"\n✅ 测试成功！请前往桌面查看 '{output_folder.name}' 文件夹。")


if __name__ == "__main__":
    run_chunk_test()