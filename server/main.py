import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import uvicorn
from fastapi import FastAPI
from routers import api_router
from server_config.database import get_db_manager
from server_config.settings import UVICORN_CONFIG
from services.agent_service import agent_manager
from graphrag_agent.models.get_models import get_embeddings_model

# 初始化 FastAPI 应用
app = FastAPI(title="知识图谱问答系统", description="基于知识图谱的智能问答系统后端API")

# 预加载嵌入模型
@app.on_event("startup")
async def startup_event():
    """服务器启动时立即加载嵌入模型到内存"""
    print(">>> 正在启动后端服务，预加载嵌入模型中...")
    try:
        # 触发单例加载
        get_embeddings_model()
        print(">>> 嵌入模型预加载完成，系统就绪。")
    except Exception as e:
        print(f">>> 模型加载失败: {str(e)}")

# 添加路由
app.include_router(api_router)

# 获取数据库连接
db_manager = get_db_manager()
driver = db_manager.driver


@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时清理资源"""
    # 关闭所有Agent资源
    agent_manager.close_all()
    
    # 关闭Neo4j连接
    if driver:
        driver.close()
        print("已关闭Neo4j连接")


# 启动服务器
if __name__ == "__main__":
    uvicorn.run("main:app", **UVICORN_CONFIG)
