import sys
from pathlib import Path
# 在文件开头添加项目根目录到系统路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st

def custom_css():
    """添加自定义CSS样式"""
    st.markdown("""
    <style>
    /* === 全局布局调整：减少留白 === */
    .main .block-container {
        padding-top: 1rem;      
        padding-bottom: 0rem;   
        padding-left: 0rem;     
        padding-right: 0rem;  
        max-width: 100%;
    }

    /* 紧凑化标题和分隔线 */
    h1 {
        padding-top: 0rem;
        margin-bottom: 0.2rem !important;
    }
    hr {
        margin-top: 0.2rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* 减少通用元素垂直间距 */
    div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.5rem;
    }

    /* 双栏独立滚动布局 */
    /* 定位主内容区的分栏 */
    div[data-testid="stHorizontalBlock"]:nth-of-type(2) {
        height: calc(100vh - 13rem);
        align-items: stretch;
        gap: 2rem;                  
        flex-wrap: nowrap !important; 
    }

    /* 设置列的内部滚动 */
    div[data-testid="stHorizontalBlock"]:nth-of-type(2) > div[data-testid="stColumn"] {
        height: 100%;
        overflow-y: auto;
        padding-bottom: 100px;
        min-width: 0;            
    }

    /* 修复列内内容容器的高度 */
    div[data-testid="stHorizontalBlock"]:nth-of-type(2) > div[data-testid="stColumn"] > div {
        height: auto;
        min-height: 100%;
    }

    /* === 美化滚动条 === */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #d1d5db;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background-color: #9ca3af;
    }

    /* === 其他原有样式 === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b9bff;
        color: white;
    }
    .agent-selector {
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 5px;
        background-color: #f7f7f7;
    }
    .chat-container {
        border-radius: 10px;
        background-color: white;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .debug-container {
        border-radius: 10px;
        background-color: white;
        padding: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .example-question {
        background-color: #f7f7f7;
        padding: 8px;
        border-radius: 4px;
        margin: 5px 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .example-question:hover {
        background-color: #e6e6e6;
    }

    /* 源内容样式 */
    .source-content-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #e0e0e0;
    }
    .source-content {
        white-space: pre-wrap;
        word-wrap: break-word;
        background-color: #f5f5f5;
        padding: 16px;
        border-radius: 4px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 14px;
        line-height: 1.6;
        overflow-x: auto;
        color: #24292e;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e1e4e8;
    }

    /* 调试信息样式 */
    .debug-header {
        background-color: #eef2f5;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 4px solid #4b9bff;
    }
    /* 1. 隐藏主容器的水平溢出，防止出现底部滚动条 */
    .main .block-container {
        overflow-x: hidden !important;
    }

    /* 2. 针对调试模式下的两栏布局，强制隐藏水平滚动条 */
    [data-testid="stHorizontalBlock"] {
        overflow-x: hidden !important;
    }

    /* 3. 针对所有 Webkit 浏览器（Chrome, Edge, Safari）彻底禁用水平滚动条显示 */
    ::-webkit-scrollbar:horizontal {
        height: 0px !important;
        display: none !important;
    }

    /* 4. 确保聊天输入框容器不会因为宽度计算偏差诱发滚动条 */
    [data-testid="stChatInput"] {
        margin-left: 1px !important;
        margin-right: 1px !important;
    }

    /* 按钮样式 */
    button:hover {
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }
    .view-source-button {
        background-color: #f1f8ff;
        border: 1px solid #c8e1ff;
        color: #0366d6;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 12px;
        margin: 4px;
    }
    .view-source-button:hover {
        background-color: #dbedff;
    }
    .processing-indicator {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 10px;
        border-radius: 4px;
        border-left: 4px solid #ffeeba;
        margin: 5px 0;
        font-size: 12px;
    }
    .iteration-round {
        background-color: #f8f9fa;
        border-left: 4px solid #4285F4;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .iteration-query {
        background-color: #f0f2f6;
        padding: 8px 12px;
        border-radius: 4px;
        font-family: monospace;
        margin: 5px 0;
    }
    .iteration-info {
        background-color: #e8f5e9;
        padding: 12px;
        border-radius: 4px;
        border-left: 3px solid #4CAF50;
        margin: 10px 0;
    }
    .iteration-progress {
        height: 8px;
        background-color: #f0f0f0;
        border-radius: 4px;
        margin: 15px 0;
        overflow: hidden;
    }
    .iteration-progress-bar {
        height: 100%;
        background-color: #4CAF50;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

KG_MANAGEMENT_CSS = """
<style>
    .kg-form {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .kg-form-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #1976d2;
    }
    .kg-entity-card {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: white;
    }
    .kg-relation-card {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: white;
    }
    .kg-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 8px;
    }
    .kg-entity-badge {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .kg-relation-badge {
        background-color: #e8f5e9;
        color: #388e3c;
    }
    .kg-property-table {
        width: 100%;
        border-collapse: collapse;
    }
    .kg-property-table th, .kg-property-table td {
        border: 1px solid #e0e0e0;
        padding: 8px;
        text-align: left;
    }
    .kg-property-table th {
        background-color: #f5f5f5;
    }
</style>
"""