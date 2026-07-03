
embeddings_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

session_id="user-001"

md5_path="./md5.txt"

collection_name="rag"

persist_directory="./chroma_db"

api_key_path = r"E:\apikey.txt"

dashscope_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

dashscope_api_key = None

chunk_overlap = 50

chunk_size = 500

separators=["\n\n","\n",",",".","?","!","，","。","？","！"," ",""]

max_split_char_number=100

similarity_threshold=1

import os

def load_api_key():
    """
    从指定路径加载 API Key
    """

    if not os.path.exists(api_key_path):
        raise FileNotFoundError(
            f"[ERROR] 未找到 API Key 文件：{api_key_path}"
        )

    with open(api_key_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("DASHSCOPE_API_KEY="):
                global dashscope_api_key
                api_key = line.strip().split("=", 1)[1].strip()
                if not api_key:
                    raise ValueError("[ERROR] DASHSCOPE_API_KEY 为空")
                os.environ["DASHSCOPE_API_KEY"] = api_key
                dashscope_api_key = api_key
                print("[INFO] 已从指定路径加载 API Key")
                return

    raise ValueError("[ERROR] 未在文件中找到 DASHSCOPE_API_KEY")


# 程序启动时自动加载（可选：也可以在 main.py 中显式调用）
load_api_key()