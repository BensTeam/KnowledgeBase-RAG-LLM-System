import os
import json
from typing import Sequence

from pathlib import Path
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, BaseMessage, messages_from_dict


def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id
        # 使用 Path 对象处理基础存储路径
        self.storage_path = Path(storage_path)

        # 构造完整的文件路径
        self.file_path = self.storage_path / f"{self.session_id}.json"

        self.storage_path.mkdir(parents=True, exist_ok=True)


    def add_messages(self,messages:Sequence[BaseMessage]):
        all_messages = list(self.messages)
        all_messages.extend(messages)

        new_messages = [message_to_dict(message) for message in all_messages]
        with open(self.file_path,'w',encoding="utf-8") as f:
            json.dump(new_messages,f)

    @property
    def messages(self):
        try:
            with open(self.file_path,'r',encoding="utf-8") as f:
                message_data = json.load(f)
                return messages_from_dict(message_data)
        except FileNotFoundError:
            return []


    def clear(self) -> None:
        with open(self.file_path,'w',encoding="utf-8") as f:
            json.dump([],f)
        