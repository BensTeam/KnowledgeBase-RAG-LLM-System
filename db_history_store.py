import os
import sqlite3
import json
from typing import List, Dict, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


class DBHistoryStore:
    """
    基于SQLite数据库的对话历史持久化存储方案
    替代原有的file_history_store文件存储方案，提供更好的性能和数据管理能力
    """

    def __init__(self, db_path: str = "./chat_history.db"):
        """
        初始化数据库连接并创建必要的表

        Args:
            db_path: SQLite数据库文件路径，默认为项目根目录下的chat_history.db
        """
        self.db_path = db_path
        self._ensure_directory_exists()
        self._init_database()

    def _ensure_directory_exists(self) -> None:
        """确保数据库文件所在目录存在"""
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _init_database(self) -> None:
        """初始化数据库表结构"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_data TEXT NOT NULL,
            message_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(create_table_sql)
                conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"数据库初始化失败: {str(e)}")

    def _create_session(self, session_id: str) -> None:
        """
        创建新的会话记录

        Args:
            session_id: 会话唯一标识符
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO chat_sessions (session_id) VALUES (?)",
                    (session_id,)
                )
                conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"创建会话失败: {str(e)}")

    def add_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        """
        添加消息到指定会话的历史记录中

        Args:
            session_id: 会话唯一标识符
            messages: 要添加的消息列表
        """
        if not messages:
            return

        try:
            self._create_session(session_id)

            with sqlite3.connect(self.db_path) as conn:
                for msg in messages:
                    msg_dict = message_to_dict(msg)
                    msg_type = msg_dict.get("type", "unknown")
                    msg_json = json.dumps(msg_dict, ensure_ascii=False)

                    conn.execute(
                        """
                        INSERT INTO chat_messages (session_id, message_data, message_type)
                        VALUES (?, ?, ?)
                        """,
                        (session_id, msg_json, msg_type)
                    )

                conn.execute(
                    "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (session_id,)
                )
                conn.commit()

        except sqlite3.Error as e:
            raise RuntimeError(f"添加消息失败: {str(e)}")

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """
        获取指定会话的所有历史消息

        Args:
            session_id: 会话唯一标识符

        Returns:
            消息列表，按创建时间排序
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT message_data FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (session_id,)
                )

                messages_data = []
                for row in cursor.fetchall():
                    msg_json = row["message_data"]
                    msg_dict = json.loads(msg_json)
                    messages_data.append(msg_dict)

            return messages_from_dict(messages_data)

        except sqlite3.Error as e:
            raise RuntimeError(f"获取消息失败: {str(e)}")

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        获取会话的基本信息

        Args:
            session_id: 会话唯一标识符

        Returns:
            会话信息字典，包含id、session_id、created_at、updated_at；不存在返回None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM chat_sessions WHERE session_id = ?",
                    (session_id,)
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"]
                    }
                return None

        except sqlite3.Error as e:
            raise RuntimeError(f"获取会话信息失败: {str(e)}")

    def list_sessions(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        获取会话列表

        Args:
            limit: 返回数量限制，默认100
            offset: 偏移量，默认0

        Returns:
            会话信息列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM chat_sessions
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset)
                )

                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"]
                    })

            return sessions

        except sqlite3.Error as e:
            raise RuntimeError(f"获取会话列表失败: {str(e)}")

    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话及其所有消息

        Args:
            session_id: 会话唯一标识符

        Returns:
            删除成功返回True，会话不存在返回False
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM chat_messages WHERE session_id = ?",
                    (session_id,)
                )
                cursor.execute(
                    "DELETE FROM chat_sessions WHERE session_id = ?",
                    (session_id,)
                )
                conn.commit()

            return cursor.rowcount > 0

        except sqlite3.Error as e:
            raise RuntimeError(f"删除会话失败: {str(e)}")

    def get_message_count(self, session_id: str) -> int:
        """
        获取指定会话的消息数量

        Args:
            session_id: 会话唯一标识符

        Returns:
            消息数量
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?",
                    (session_id,)
                )

                return cursor.fetchone()[0]

        except sqlite3.Error as e:
            raise RuntimeError(f"获取消息数量失败: {str(e)}")


# 全局实例，方便其他模块使用
_history_store = DBHistoryStore()


def get_history(session_id: str) -> DBHistoryStore:
    """
    获取对话历史存储实例

    Args:
        session_id: 会话唯一标识符

    Returns:
        DBHistoryStore实例
    """
    return _history_store


if __name__ == "__main__":
    """测试代码"""
    from langchain_core.messages import HumanMessage, AIMessage

    store = DBHistoryStore()

    test_session_id = "test_session_001"

    store.add_messages(test_session_id, [
        HumanMessage(content="你好"),
        AIMessage(content="您好！我是您的智能助手。"),
        HumanMessage(content="我体重180斤，穿什么尺码？")
    ])

    messages = store.get_messages(test_session_id)
    print(f"会话 {test_session_id} 的消息数量: {len(messages)}")
    for i, msg in enumerate(messages):
        print(f"{i+1}. {msg.type}: {msg.content}")

    sessions = store.list_sessions()
    print(f"\n所有会话: {len(sessions)}")
    for sess in sessions:
        print(f"  - {sess['session_id']}")

    store.delete_session(test_session_id)
    print(f"\n删除后会话列表: {len(store.list_sessions())}")
