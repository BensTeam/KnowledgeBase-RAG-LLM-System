import time

import streamlit as st
from langchain_community.tools.connery import service
from milvus_lite.adapter.grpc.translators import result

from knowledge_base import KnowledgeBaseService


#添加标题
st.title("知识库更新服务")

if "service" not in st.session_state:
    st.session_state.service = KnowledgeBaseService()

service = st.session_state.service

#上传控件
uploader_file=st.file_uploader(
    "请上传TXT文件",
    type="txt",
    accept_multiple_files=False,
)


if uploader_file is not None:
    file_name=uploader_file.name
    file_type=uploader_file.type
    file_size=uploader_file.size/1024

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type}|大小：{file_size:.2f}KB")
    text=uploader_file.getvalue().decode("utf-8")

    with st.spinner("载入知识库中..."):
        time.sleep(1)
        result=service.upload_by_str(text, file_name)
        st.write(result)