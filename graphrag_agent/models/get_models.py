import sys
from pathlib import Path

# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
# 新增：引入本地模型支持
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os

from graphrag_agent.config.settings import (
    TIKTOKEN_CACHE_DIR,
    OPENAI_EMBEDDING_CONFIG,
    OPENAI_LLM_CONFIG,
)


# === 全局单例缓存变量 ===
_embeddings_instance = None
_llm_instance = None
_stream_llm_instance = None


# 设置 tiktoken 缓存目录，避免每次联网拉取
def setup_cache():
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)


setup_cache()


def get_embeddings_model():
    """
    获取 Embedding 模型（单例模式）。
    确保全局只加载一次，显著降低 fast_cache_check 耗时。
    """
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    model_name = OPENAI_EMBEDDING_CONFIG.get("model", "")

    # 判断是否为本地模型
    if model_name and (os.path.sep in model_name or "/" in model_name or os.path.exists(model_name)):
        print(f"首次加载本地 Embedding 模型: {model_name}...")

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        # OpenAI/Ollama 逻辑
        config = {k: v for k, v in OPENAI_EMBEDDING_CONFIG.items() if v}
        _embeddings_instance = OpenAIEmbeddings(**config)

    return _embeddings_instance


def get_llm_model():
    """获取 LLM 模型（单例模式）"""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    config = {k: v for k, v in OPENAI_LLM_CONFIG.items() if v is not None and v != ""}
    _llm_instance = ChatOpenAI(**config)
    return _llm_instance


def get_stream_llm_model():
    """获取流式 LLM 模型（单例模式）"""
    global _stream_llm_instance
    if _stream_llm_instance is not None:
        return _stream_llm_instance

    callback_handler = AsyncIteratorCallbackHandler()
    manager = AsyncCallbackManager(handlers=[callback_handler])

    config = {k: v for k, v in OPENAI_LLM_CONFIG.items() if v is not None and v != ""}
    config.update({"streaming": True, "callbacks": manager})
    _stream_llm_instance = ChatOpenAI(**config)
    return _stream_llm_instance


def count_tokens(text):
    """简单通用的token计数"""
    if not text:
        return 0

    model_name = (OPENAI_LLM_CONFIG.get("model") or "").lower()

    # 如果是deepseek，使用transformers
    if 'deepseek' in model_name:
        try:
            from transformers import AutoTokenizer
            # 这里也可以尝试读取本地路径，如果没有则默认联网
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
            return len(tokenizer.encode(text))
        except:
            pass

    # 如果是gpt，使用tiktoken
    if 'gpt' in model_name:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            pass

    # 备用方案：简单计算
    chinese = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english = len(text) - chinese
    return chinese + english // 4



if __name__ == '__main__':
    # 测试llm
    llm = get_llm_model()
    print(llm.invoke("你好"))

    # 由于langchain版本问题，这个目前测试会报错
    # llm_stream = get_stream_llm_model()
    # print(llm_stream.invoke("你好"))

    # 测试embedding
    test_text = "你好，这是一个测试。"
    embeddings = get_embeddings_model()
    print(embeddings.embed_query(test_text))

    # 测试计数
    test_text = "Hello 你好世界"
    tokens = count_tokens(test_text)
    print(f"Token计数: '{test_text}' = {tokens} tokens")
