import sys
import re
from pathlib import Path

# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

import hanlp
import re
from typing import List, Tuple, Optional

# 保留原仓库的配置引用，维持项目一致性
from graphrag_agent.config.settings import CHUNK_SIZE, OVERLAP, MAX_TEXT_LENGTH


class ChineseTextChunker:
    """中文文本分块器：在指定窗口内按句末标点优先截断"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP, max_text_length: int = MAX_TEXT_LENGTH):
        """
        初始化分块器

        Args:
            chunk_size: 每个文本块的目标大小（tokens数量）
            overlap: 相邻文本块的重叠大小（tokens数量）
            max_text_length: (保留参数以兼容旧接口，新逻辑主要依赖流式处理)
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size必须大于overlap")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_text_length = max_text_length
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

        # 句末标点集合：中英文、全角半角
        self._sentence_endings = {
            '。', '！', '？', '!', '?', '．', '.'
        }

    def process_files(self, file_contents: List[Tuple[str, str]]) -> List[Tuple[str, str, List[List[str]]]]:
        """
        处理多个文件的内容
        """
        results = []
        for filename, content in file_contents:
            chunks = self.chunk_text(content)
            results.append((filename, content, chunks))
        return results

    def chunk_text(self, text: str) -> List[List[str]]:
        """
        将单个文本分割成块（逻辑已修改为本地文件的 Buffer 模式，并修复了死循环bug）
        """
        # 处理空文本或太短的文本
        if not text or len(text) < self.chunk_size / 5:
            try:
                tokens = self.tokenizer(text)
                return [tokens] if tokens else []
            except Exception as e:
                print(f"分词失败：{str(e)}")
                return [text.split()] if text else []

        parts = re.split(r'(\n+)', text)
        buffer: List[str] = []
        chunks: List[List[str]] = []

        # 定义弹性窗口大小：
        target_max_size = self.chunk_size + 100

        part_idx = 0
        while part_idx < len(parts):
            while len(buffer) < target_max_size and part_idx < len(parts):
                p = parts[part_idx]
                if not p:
                    part_idx += 1
                    continue
                if '\n' in p:
                    token = '\n\n' if len(p) >= 2 else '\n'
                    buffer.append(token)
                else:
                    try:
                        tokens = self.tokenizer(p)
                        buffer.extend(tokens)
                    except Exception:
                        buffer.extend(p.split())
                part_idx += 1

            while len(buffer) >= self.chunk_size:
                # 寻找边界 (min_size 和 max_size 传入新计算的值)
                end_idx = self._find_chunk_boundary(buffer, self.chunk_size - 100, target_max_size)

                chunk = buffer[:end_idx]
                chunks.append(chunk)

                # 【核心修改】：没有重叠的部分，直接截断 buffer
                # 强制步进，防止 end_idx 为 0 时死循环
                buffer = buffer[max(1, end_idx):]

        if buffer:
            chunks.append(buffer)
        return chunks

    # ================= 工具方法 =================

    def _is_sentence_end(self, token: str) -> bool:
        """判断token是否为句子结束符"""
        return token in self._sentence_endings

    def _find_next_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """从指定位置向后查找句子结束位置"""
        for i in range(start_pos, len(tokens)):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return len(tokens)

    def _find_previous_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """从指定位置向前查找句子结束位置"""
        # 注意边界检查
        if start_pos >= len(tokens):
            start_pos = len(tokens) - 1

        for i in range(start_pos, -1, -1):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return 0

    def tokens_to_text(self, tokens) -> str:
        """把分词结果拼回字符串，保留英文单词间的空格"""
        if not tokens:
            return ""

        # 预先编译正则表达式以提高效率
        ascii_pattern = re.compile(r'^[a-zA-Z0-9]+$')

        result = []
        prev_is_ascii = False

        for token in tokens:
            # 使用正则表达式快速检查是否为纯ASCII字母数字
            current_is_ascii = bool(ascii_pattern.match(token))

            # 如果前一个token是ASCII且当前也是ASCII，则在它们之间添加空格
            if prev_is_ascii and current_is_ascii:
                result.append(' ')

            result.append(token)
            prev_is_ascii = current_is_ascii

        return ''.join(result)

    def _find_chunk_boundary(self, tokens: List[str], min_size: int, max_size: int) -> int:
        """
        按照自定义范围和优先级寻找边界
        """
        n = len(tokens)
        # 定义关键边界点
        limit_300 = max(0, self.chunk_size - 100)
        limit_400 = self.chunk_size
        limit_500 = min(n, self.chunk_size + 100)

        # 1. 优先级：连续两个换行符 \n\n 在 [300, 500] 之间
        for i in range(limit_300, limit_500 + 1):
            if i - 1 < n and tokens[i - 1] == '\n\n':
                return i

        # 2. 优先级：单个换行符 \n 在 [400, 500] 之间
        # 从后往前找，尽量保留更多内容
        for i in range(limit_500, limit_400 - 1, -1):
            if i - 1 < n and tokens[i - 1] == '\n':
                return i

        # 3. 优先级：单个换行符 \n 在 [300, 400] 之间
        for i in range(limit_400, limit_300 - 1, -1):
            if i - 1 < n and tokens[i - 1] == '\n':
                return i

        # 4. 优先级：标点符号在 [400, 500] 之间
        for i in range(limit_500, limit_400 - 1, -1):
            if i - 1 < n and self._is_sentence_end(tokens[i - 1]):
                return i

        # 5. 最终兜底：返回整个文本 (buffer 中目前所有的 tokens)
        return n