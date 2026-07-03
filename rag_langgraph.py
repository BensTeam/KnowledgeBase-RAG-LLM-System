"""
RAG服务 - LangGraph版本
基于LangGraph框架重构的RAG服务，使用SQLite数据库持久化对话历史

特性：
- 使用LangGraph替代已弃用的RunnableWithMessageHistory
- 使用SQLite数据库持久化对话历史（db_history_store）
- 模块化设计，代码结构清晰
- 完整的错误处理机制
- 支持多会话管理
- 符合行业最佳实践
"""

import logging
from typing import TypedDict, Annotated
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from vector_store import VectorStoreService
from MyDashScopeEmbeddings import DashScopeEmbeddings
from db_history_store import DBHistoryStore
import config_data as config


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class RagState(TypedDict):
    """
    RAG工作流的状态定义
    使用TypedDict定义状态结构，确保类型安全
    """
    input: str
    context: str
    messages: Annotated[list[BaseMessage], add]


class RagService:
    """
    RAG服务类
    提供基于LangGraph的检索增强生成功能
    """

    def __init__(self):
        """
        初始化RAG服务

        加载配置、初始化向量存储、聊天模型、数据库存储和LangGraph工作流
        """
        logger.info("初始化RAG服务...")

        try:
            # 初始化数据库存储服务
            self._init_history_store()

            # 初始化向量存储服务
            self._init_vector_store()

            # 初始化聊天模型
            self._init_chat_model()

            # 初始化提示模板
            self._init_prompt_template()

            # 构建LangGraph工作流
            self.chain = self._build_workflow()

            logger.info("RAG服务初始化完成")

        except Exception as e:
            logger.error(f"RAG服务初始化失败: {str(e)}", exc_info=True)
            raise

    def _init_history_store(self) -> None:
        """
        初始化数据库存储服务
        使用SQLite数据库持久化对话历史，替代内存存储
        """
        logger.debug("初始化数据库存储服务...")
        self.history_store = DBHistoryStore()

    def _init_vector_store(self) -> None:
        """
        初始化向量存储服务
        使用自定义的DashScopeEmbeddings作为嵌入模型
        """
        logger.debug("初始化向量存储服务...")
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings()
        )
        self.retriever = self.vector_service.get_retriever()

    def _init_chat_model(self) -> None:
        """
        初始化聊天模型
        使用langchain_openai的ChatOpenAI通过兼容模式调用通义千问
        """
        logger.debug("初始化聊天模型...")
        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
            temperature=0.7
        )

    def _init_prompt_template(self) -> None:
        """
        初始化提示模板
        包含系统提示、对话历史占位符和用户输入
        """
        logger.debug("初始化提示模板...")
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    你是一个专业的智能助手，基于提供的参考资料回答用户问题。

                    参考资料：
                    {context}

                    请遵循以下规则：
                    1. 回答必须基于提供的参考资料，不要编造信息
                    2. 如果参考资料中没有相关信息，请明确说明
                    3. 回答要简洁、专业、易懂
                    4. 如果有多个答案，请列出所有可能的选项
                    """
                ),
                MessagesPlaceholder("messages"),
                ("user", "{input}")
            ]
        )

    @staticmethod
    def _format_documents(docs: list[Document]) -> str:
        """
        格式化检索到的文档

        Args:
            docs: 文档列表

        Returns:
            格式化后的文档字符串
        """
        if not docs:
            return "无参考资料"

        format_str = ""
        for i, doc in enumerate(docs, 1):
            format_str += f"参考资料 {i}：\n"
            format_str += f"内容：{doc.page_content}\n"
            if doc.metadata:
                format_str += f"元数据：{doc.metadata}\n"
            format_str += "\n"

        return format_str.strip()

    def _retrieve(self, state: RagState) -> dict[str, str]:
        """
        检索节点：从向量数据库中检索相关文档

        Args:
            state: 当前状态，包含用户输入

        Returns:
            更新后的状态，包含检索到的上下文
        """
        try:
            logger.info(f"开始检索，输入: {state['input'][:50]}...")

            # 从向量数据库检索相关文档
            docs = self.retriever.invoke(state["input"])

            # 格式化文档为上下文字符串
            context = RagService._format_documents(docs)

            logger.info(f"检索完成，找到 {len(docs)} 条相关文档")

            return {"context": context}

        except Exception as e:
            logger.error(f"检索失败: {str(e)}", exc_info=True)
            return {"context": "检索失败，请稍后重试"}

    def _generate(self, state: RagState) -> dict[str, list[AIMessage]]:
        """
        生成节点：基于上下文和对话历史生成回答

        Args:
            state: 当前状态，包含用户输入、上下文和对话历史

        Returns:
            更新后的状态，包含生成的AI消息
        """
        try:
            logger.info("开始生成回答...")

            # 构建完整的提示词，bind 方法返回一个 Runnable
            prompt_value = self.prompt_template.invoke(
                {
                    "input": state["input"],
                    "context": state["context"],
                    "messages": state["messages"]
                }
            )

            # 调用大模型生成回答
            response = self.chat_model.invoke(prompt_value)

            # 将响应转换为AIMessage
            ai_message = AIMessage(content=response.content)

            logger.info(f"生成完成，回答长度: {len(ai_message.content)}")

            return {"messages": [ai_message]}

        except Exception as e:
            logger.error(f"生成失败: {str(e)}", exc_info=True)
            error_message = AIMessage(content=f"生成回答时发生错误: {str(e)}")
            return {"messages": [error_message]}

    def _build_workflow(self) -> CompiledStateGraph:
        """
        构建LangGraph工作流

        创建包含检索和生成两个节点的有向图，使用DBHistoryStore持久化对话历史

        Returns:
            编译后的LangGraph应用
        """
        logger.debug("构建LangGraph工作流...")

        # 创建状态图
        workflow: StateGraph[RagState] = StateGraph(RagState) # type: ignore[arg-type]

        # 添加节点
        workflow.add_node("retrieve", self._retrieve)  # type: ignore[arg-type]
        workflow.add_node("generate", self._generate)  # type: ignore[arg-type]

        # 设置入口点和边
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # 不使用LangGraph的checkpointer，使用自定义的DBHistoryStore管理历史
        # 这样可以更好地控制对话历史的存储和检索
        app = workflow.compile(checkpointer=None)

        return app

    def invoke(self, input_text: str, session_id: str) -> str:
        """
        调用RAG服务生成回答

        Args:
            input_text: 用户输入文本
            session_id: 会话唯一标识符

        Returns:
            AI生成的回答文本
        """
        try:
            logger.info(f"处理会话 {session_id} 的请求: {input_text[:50]}...")

            # 从数据库加载历史消息
            history_messages: list[BaseMessage] = self.history_store.get_messages(session_id)
            logger.debug(f"加载历史消息: {len(history_messages)} 条")

            # 创建用户消息
            user_message = HumanMessage(content=input_text)

            # 调用LangGraph工作流，传入用户输入、空上下文（检索时会自动填充）和历史消息
            initial_state: RagState = {
                "input": input_text,
                "context": "",
                "messages": history_messages
            }
            result = self.chain.invoke(initial_state)

            # 获取生成的AI消息
            ai_messages: list[BaseMessage] = result.get("messages", [])
            if not ai_messages:
                logger.warning("未找到有效回答")
                return "抱歉，我无法回答这个问题。"

            ai_message = ai_messages[-1]
            if not isinstance(ai_message, AIMessage):
                logger.warning(f"预期AIMessage，实际类型: {type(ai_message).__name__}")
                return "抱歉，我无法回答这个问题。"

            # 将用户消息和AI消息保存到数据库
            self.history_store.add_messages(session_id, [user_message, ai_message])
            logger.debug("消息已保存到数据库")

            answer: str = str(ai_message.content)
            logger.info(f"回答生成成功: {answer[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"调用RAG服务失败: {str(e)}", exc_info=True)
            return f"服务暂时不可用，请稍后重试。错误信息: {str(e)}"


def main():
    """
    主函数：演示RAG服务的使用
    """
    try:
        # 创建RAG服务实例
        rag_service = RagService()

        # 模拟对话
        session_id = "user_001"

        # 第一次对话
        print("=" * 50)
        print("第一次对话")
        print("=" * 50)
        question1 = "我体重180斤，穿什么尺码？"
        print(f"用户: {question1}")
        answer1 = rag_service.invoke(question1, session_id)
        print(f"AI: {answer1}")

        # 第二次对话（带上下文）
        print("\n" + "=" * 50)
        print("第二次对话（带上下文）")
        print("=" * 50)
        question2 = "那我身高190CM呢？"
        print(f"用户: {question2}")
        answer2 = rag_service.invoke(question2, session_id)
        print(f"AI: {answer2}")

        # 新会话
        print("\n" + "=" * 50)
        print("新会话")
        print("=" * 50)
        question3 = "什么是RAG？"
        print(f"用户: {question3}")
        answer3 = rag_service.invoke(question3, "new_user_002")
        print(f"AI: {answer3}")

    except Exception as e:
        logger.error(f"主程序执行失败: {str(e)}", exc_info=True)
        print(f"程序执行失败: {str(e)}")


if __name__ == "__main__":
    main()
