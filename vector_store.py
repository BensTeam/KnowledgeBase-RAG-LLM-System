import config_data as config

from langchain_chroma import Chroma


class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding=embedding

        self.vector_store=Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})


if __name__ == '__main__':
    from MyDashScopeEmbeddings import DashScopeEmbeddings
    retriever=VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res=retriever.invoke("我体重是180斤，推荐一个尺码")
    print(res)