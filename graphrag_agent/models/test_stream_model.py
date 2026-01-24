import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

import asyncio
from langchain_core.messages import HumanMessage
from graphrag_agent.models.get_models import get_stream_llm_model

async def main():
    chat = get_stream_llm_model() 
    messages = [HumanMessage(content="Tell me a short joke.")]
    try:
        async for chunk in chat.astream(messages):
            print(chunk.content, end="", flush=True)
        print("\nStream finished.")
    except Exception as e:
        print(f"\nError during basic stream: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())