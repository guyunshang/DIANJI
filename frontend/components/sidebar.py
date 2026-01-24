import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
from utils.api import clear_chat
from frontend_config.settings import examples

def display_sidebar():
    """显示应用侧边栏"""
    with st.sidebar:
        st.title("📚 GraphRAG")
        st.markdown("---")

        # Agent选择部分
        st.header("Agent：hybrid_agent")
        st.session_state.agent_type = "hybrid_agent"


        st.markdown("---")

        # 系统设置部分 - 组合调试模式和响应设置
        st.header("系统设置")

        # 调试选项
        debug_mode = st.checkbox("启用调试模式",
                               value=st.session_state.debug_mode,
                               key="sidebar_debug_mode",
                               help="显示执行轨迹、知识图谱和源内容")

        # 当调试模式切换时，处理流式响应状态
        previous_debug_mode = st.session_state.debug_mode
        if debug_mode != previous_debug_mode:
            if debug_mode:
                # 启用调试模式时，禁用流式响应
                st.session_state.use_stream = False

        # 更新全局debug_mode
        st.session_state.debug_mode = debug_mode

        # 添加流式响应选项（仅当调试模式未启用时显示）
        if not debug_mode:
            use_stream = st.checkbox("使用流式响应",
                                   value=st.session_state.get("use_stream", True),
                                   key="sidebar_use_stream",
                                   help="启用流式响应，实时显示生成结果")
            # 更新全局 use_stream
            st.session_state.use_stream = use_stream
        else:
            # 在调试模式下显示提示
            st.info("调试模式下已禁用流式响应")

        st.markdown("---")

        # 示例问题部分
        st.header("示例问题")
        example_questions = examples

        for question in example_questions:
            st.markdown(f"""
            <div style="background-color: #f7f7f7; padding: 8px; 
                 border-radius: 4px; margin: 5px 0; font-size: 14px; cursor: pointer;">
                {question}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 项目信息
        st.markdown("""
        ### 关于
        这个 GraphRAG 演示基于本地文档建立的知识图谱，可以使用不同的Agent策略回答问题。
        
        **调试模式**可查看:
        - 执行轨迹
        - 知识图谱可视化
        - 原始文本内容
        - 性能监控
        """)

        # 重置按钮
        if st.button("🗑️ 清除对话历史", key="clear_chat"):
            clear_chat()