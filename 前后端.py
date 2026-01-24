import subprocess
import sys
import os
import time
from pathlib import Path

# 设置项目根目录路径
# 脚本将根据你提供的路径运行，确保工作目录正确
PROJECT_ROOT = Path(r"F:\Pycharm\PycharmProjects\TAO\graph-rag-agent-new")


def start_all_services():
    """同时启动后端和前端服务"""

    # 切换当前工作目录到项目根目录
    os.chdir(PROJECT_ROOT)
    print(f"当前工作目录: {os.getcwd()}")

    # 1. 启动后端服务器 (FastAPI)
    # 根据项目结构，入口为 server/main.py
    print("正在启动后端服务器 (server/main.py)...")
    backend_cmd = [sys.executable, "server/main.py"]
    backend_process = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT)

    # 稍等几秒，确保后端完成初始化（如连接 Neo4j 数据库）
    time.sleep(5)

    # 2. 启动前端应用 (Streamlit)
    # 根据项目结构，入口为 frontend/app.py
    print("正在启动前端应用 (frontend/app.py)...")
    frontend_cmd = ["streamlit", "run", "frontend/app.py"]
    frontend_process = subprocess.Popen(frontend_cmd, cwd=PROJECT_ROOT)

    print("\n" + "=" * 50)
    print("系统已启动！")
    print("后端进程 ID:", backend_process.pid)
    print("前端进程 ID:", frontend_process.pid)
    print("提示: 按下 Ctrl+C 可同时停止所有服务。")
    print("=" * 50 + "\n")

    try:
        # 持续监控子进程状态
        while True:
            time.sleep(1)
            # 检查子进程是否意外退出
            if backend_process.poll() is not None:
                print("警告: 后端服务已停止。")
                break
            if frontend_process.poll() is not None:
                print("警告: 前端服务已停止。")
                break
    except KeyboardInterrupt:
        print("\n接收到停止指令，正在关闭服务...")
    finally:
        # 确保脚本退出时关闭两个子进程
        backend_process.terminate()
        frontend_process.terminate()
        print("服务已安全停止。")


if __name__ == "__main__":
    start_all_services()