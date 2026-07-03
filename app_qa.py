from time import sleep

import streamlit as st

from rag_langgraph import RagService
import config_data as config

st.title("智能客服")
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"你好有什么可以帮到你？"}]

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

if "rag_service" not in st.session_state:
    st.session_state["rag_service"]=RagService()

rag_service=st.session_state["rag_service"]

prompt= st.chat_input()

if prompt :
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role":"user","content":prompt})

    with st.spinner("AI思考中..."):
        answer=rag_service.invoke(prompt,config.session_id)
        st.chat_message("ai").write(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})