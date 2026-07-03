import os
import hashlib
import config_data as config
from MyDashScopeEmbeddings import DashScopeEmbeddings
from datetime import datetime
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter



def check_md5(md5_str: str) -> bool:
    if not os.path.exists(config.md5_path):
        return False
    with open(config.md5_path, "r", encoding="utf-8") as f:
        return md5_str in {line.strip() for line in f}


def save_md5(md5_str: str):
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding="utf-8") -> str:
    return hashlib.md5(input_str.encode(encoding)).hexdigest()


class KnowledgeBaseService:
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            embedding_function=DashScopeEmbeddings(),
            persist_directory=config.persist_directory,
            collection_name=config.collection_name
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len
        )

    def upload_by_str(self, data: str, filename):
        md5_hex=get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已存在知识库中"

        if len(data) > config.max_split_char_number:
            knowledge_chunk=self.splitter.split_text(data)
        else:
            knowledge_chunk=[data]

        self.chroma.add_texts(
            texts=knowledge_chunk,
            metadatas=[{
                "source": filename,
                "creat_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operator":"Ben"
            }] * len(knowledge_chunk)
        )

        save_md5(md5_hex)

        return "[成功]内容已经成功载入向量库"


if __name__ == "__main__":
    kb = KnowledgeBaseService()
    r=kb.upload_by_str("RAG 是检索增强生成技术", filename="test.txt")
    print(r)
    result = kb.chroma.similarity_search("什么是 RAG？", k=1)
    print(result)