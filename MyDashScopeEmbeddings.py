#基于Embeddings，自行封装DashScopeEmbeddings，解决langchain-community停止维护的问题
from dashscope import embeddings, Generation
from langchain_core.embeddings import Embeddings

import config_data as config


class DashScopeEmbeddings(Embeddings):
    def __init__(self, model: str = config.embeddings_model_name):
        self.model = model
        self.api_key = config.dashscope_api_key

    def embed_documents(self, texts):
        embedding_result = embeddings.TextEmbedding.call(
            model=self.model,
            input=texts,
            api_key=self.api_key
        )
        return [item["embedding"] for item in embedding_result.output["embeddings"]]

    def embed_query(self, text):
        query_result = embeddings.TextEmbedding.call(
            model=self.model,
            input=[text],
            api_key=self.api_key
        )
        return query_result.output["embeddings"][0]["embedding"]

