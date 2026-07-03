# RAG 项目 - 检索增强生成系统

> 基于 LangChain 和 LangGraph 构建的智能问答系统，支持多会话管理和对话历史持久化
<img width="753" height="913" alt="image" src="https://github.com/user-attachments/assets/89d318cf-ce01-455b-8c74-9ce77503d1d5" />


---

## 项目简介

本项目是一个基于 **RAG（Retrieval-Augmented Generation）** 技术的智能问答系统，能够根据用户的提问从知识库中检索相关文档，并结合大语言模型生成准确、专业的回答。

### 主要功能

- ✅ 知识库文档检索
- ✅ 多轮对话支持
- ✅ 对话历史持久化
- ✅ 多会话管理
- ✅ 错误处理和日志记录

---

## 技术栈

| 组件 | 技术 | 版本 |
|-----|------|------|
| 框架 | LangChain / LangGraph | 1.x |
| 向量数据库 | ChromaDB | 1.5.x |
| 嵌入模型 | DashScope Embedding | text-embedding-v4 |
| 聊天模型 | 通义千问 | qwen3-max |
| 数据库 | SQLite | 内置 |
| 语言 | Python | 3.12+ |

---

## 版本对比

### 老版本（v1.0）问题

| 问题 | 描述 | 严重程度 |
|-----|------|---------|
| 弃用警告 | `langchain-community` 包已停止维护 | 中 |
| 内存存储 | 对话历史仅存储在内存中，重启丢失 | 高 |
| 文件存储 | 使用文件存储对话历史，性能差 | 中 |
| 状态管理 | `RunnableWithMessageHistory` 已弃用 | 中 |


### 新版本（v1.1）改进

| 改进项 | 老版本 | 新版本 | 优势 |
|-------|-------|-------|------|
| **嵌入模型** | `langchain_community.DashScopeEmbeddings` | 自定义 `MyDashScopeEmbeddings` | 避免弃用警告，直接调用原生 SDK |
| **聊天模型** | `langchain_community.ChatTongyi` | `langchain_openai.ChatOpenAI` + 兼容模式 | 官方维护，功能完整 |
| **对话历史** | 文件存储 / 内存存储 | SQLite 数据库存储 | 持久化、高性能、支持索引 |
| **状态管理** | `RunnableWithMessageHistory` | LangGraph 框架 | 官方推荐，支持复杂工作流 |
| **API Key** | 环境变量 | 模块变量 | 多线程环境下可靠传递 |

---

## 项目结构

```
RAG/
├── config_data.py          # 配置文件（API Key、模型名称等）
├── MyDashScopeEmbeddings.py # 自定义嵌入模型
├── vector_store.py         # 向量存储服务
├── knowledge_base.py       # 知识库管理
├── rag.py                  # 旧版 RAG 服务（兼容保留）
├── rag_langgraph.py        # 新版 RAG 服务（推荐使用）
├── file_history_store.py   # 旧版文件存储（兼容保留）
├── db_history_store.py     # 新版数据库存储（推荐使用）
├── chat_history.db         # SQLite 数据库文件（自动生成）
├── chroma_db/              # ChromaDB 向量数据库
└── requirements.txt        # 依赖列表
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

创建 API Key 文件（如 `E:\apikey.txt`），内容格式：

```
DASHSCOPE_API_KEY=your_api_key_here
```

修改 `config_data.py` 中的 `api_key_path` 指向该文件：

```python
api_key_path = r"E:\apikey.txt"
```

### 3. 运行服务

使用新版 LangGraph 服务（推荐）：

```bash
python rag_langgraph.py
```

使用旧版服务（兼容保留）：

```bash
python rag.py
```

### 4. 使用示例

```python
from rag_langgraph import RagService

# 创建服务实例
rag_service = RagService()

# 调用服务
result = rag_service.invoke("我体重180斤，穿什么尺码？", "user_001")
print(result)

# 继续对话（自动带上历史上下文）
result = rag_service.invoke("那我身高190CM呢？", "user_001")
print(result)
```

---

## 核心模块说明

### rag_langgraph.py - 新版 RAG 服务

基于 **LangGraph** 框架构建的 RAG 服务，包含：

- **RagState**：TypedDict 状态定义，包含输入、上下文和消息历史
- **RagService**：RAG 服务类，提供检索和生成功能
- **工作流**：`retrieve`（检索）→ `generate`（生成）→ `END`

### db_history_store.py - 数据库存储

基于 **SQLite** 的对话历史存储方案，提供：

- **chat_sessions**：会话信息表
- **chat_messages**：消息记录表
- **CRUD 操作**：添加、获取、列表、删除会话

### MyDashScopeEmbeddings.py - 自定义嵌入模型

基于 **DashScope SDK** 封装的嵌入模型，避免依赖已弃用的 `langchain-community`。

---

## 关键技术要点

### 1. ChatOpenAI 兼容模式

使用 `langchain_openai.ChatOpenAI` 通过兼容模式调用通义千问：

```python
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    model="qwen3-max",
    api_key=config.dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

### 2. LangGraph 工作流

使用 LangGraph 定义状态驱动的工作流：

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(RagState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile(checkpointer=None)  # 使用自定义DBHistoryStore
```

**关于 checkpointer=None**：
本项目使用自定义的 `DBHistoryStore`（SQLite）管理对话历史，而不是 LangGraph 内置的 checkpointer。这样做的优势是：
- 更好地控制对话历史的存储格式和检索逻辑
- 便于与其他系统集成（如 Web 界面）
- 支持更灵活的会话管理功能（如删除、统计）

### 3. SQLite 持久化

使用 SQLite 数据库持久化对话历史：

```python
from db_history_store import DBHistoryStore

store = DBHistoryStore()
store.add_messages(session_id, [user_message, ai_message])
messages = store.get_messages(session_id)
```

### 4. 自定义嵌入模型

由于 `https://dashscope.aliyuncs.com/compatible-mode/v1` 域名**不支持 Embedding API**，因此需要使用自定义的 `MyDashScopeEmbeddings` 直接调用 DashScope SDK：

```python
from MyDashScopeEmbeddings import DashScopeEmbeddings

embedding = DashScopeEmbeddings()  # 直接调用原生SDK
```

> **注意**：如果使用阿里云百炼专属域名（如 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com`），可以使用 `langchain_openai.OpenAIEmbeddings` 替代自定义实现。自定义实现主要是为了让朋友们更容易理解Embeddings模型的使用。

---

## 增加运行日志

新版服务提供完整的日志记录：

```
2026-07-02 15:52:54,038 - __main__ - INFO - 初始化RAG服务...
2026-07-02 15:52:54,834 - __main__ - INFO - RAG服务初始化完成
2026-07-02 15:52:54,834 - __main__ - INFO - 处理会话 user_001 的请求: 我体重180斤...
2026-07-02 15:52:55,375 - __main__ - INFO - 检索完成，找到 1 条相关文档
2026-07-02 15:52:58,567 - httpx - INFO - HTTP Request: POST https://dashscope.aliyuncs.com...
2026-07-02 15:52:58,604 - __main__ - INFO - 回答生成成功: 根据参考资料...
```

