
import config_data as config
from file_history_store import get_history
from vector_store import VectorStoreService
from MyDashScopeEmbeddings import DashScopeEmbeddings


from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser


class RagService:
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings()
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户的问题。参考资料：{context}"),
                ("system","并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户的提问：{input}")
            ]
        )

        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无参考资料"
            format_str = ""
            for doc in docs:
                format_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return format_str

        def showprompt(prompt_value):
            print("*"*20)
            print(prompt_value.to_string())
            print("*" * 20)
            return prompt_value

        def temp1(prompt_value):
            return prompt_value["input"]

        def temp2(prompt_value):
            return {"input":prompt_value["input"]["input"],
                    "history":prompt_value["input"]["history"],
                    "context":prompt_value["context"]
                    }

        chain = (
            RunnableMap(
                input=RunnablePassthrough(),
                context= RunnableLambda(temp1)
                         | retriever
                         | RunnableLambda(format_document)
            )
            |RunnableLambda(temp2)
            | self.prompt_template
            #| RunnableLambda(showprompt)          #检查完整提示词
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chian=RunnableWithMessageHistory(
            chain,
            get_session_history=get_history,
            input_messages_key="input",
            history_messages_key="history"
        )


        return conversation_chian


if __name__ == '__main__':
    session_config={
        "configurable":{
            "session_id":"user_001",
        }
    }

    res = RagService().chain.invoke({"input":"我190CM，穿什么颜色的衣服？"},session_config)

    #res = RagService().chain.invoke({"input": "我190CM，穿什么颜色的衣服？"},
                                    #config={"configurable": {"session_id": "user-001"}})
    print(res)