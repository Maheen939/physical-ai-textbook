"""
Neon Postgres Service for conversation history
"""

import os
from typing import List, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

class NeonService:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Create conversations table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        chapter_context VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create messages table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_user_id
                    ON conversations(user_id)
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                    ON messages(conversation_id)
                """)

                print("Database schema initialized successfully")

    def create_conversation(
        self,
        user_id: str,
        chapter_context: Optional[str] = None
    ) -> int:
        """
        Create a new conversation

        Args:
            user_id: User identifier
            chapter_context: Current chapter context

        Returns:
            Conversation ID
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversations (user_id, chapter_context)
                    VALUES (%s, %s)
                    RETURNING id
                """, (user_id, chapter_context))

                result = cur.fetchone()
                return result['id']

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str
    ):
        """
        Add a message to a conversation

        Args:
            conversation_id: Conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO messages (conversation_id, role, content)
                    VALUES (%s, %s, %s)
                """, (conversation_id, role, content))

                # Update conversation timestamp
                cur.execute("""
                    UPDATE conversations
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (conversation_id,))

    def get_conversation_history(
        self,
        conversation_id: int,
        limit: int = 50
    ) -> List[dict]:
        """
        Get conversation message history

        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to retrieve

        Returns:
            List of message dicts with role, content, created_at
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT role, content, created_at
                    FROM messages
                    WHERE conversation_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                """, (conversation_id, limit))

                return cur.fetchall()

    def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[dict]:
        """
        Get all conversations for a user

        Args:
            user_id: User identifier
            limit: Maximum number of conversations

        Returns:
            List of conversation dicts
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        c.id,
                        c.chapter_context,
                        c.created_at,
                        c.updated_at,
                        COUNT(m.id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE c.user_id = %s
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    LIMIT %s
                """, (user_id, limit))

                return cur.fetchall()

    def get_or_create_conversation(
        self,
        user_id: str,
        chapter_context: Optional[str] = None
    ) -> int:
        """
        Get existing conversation or create new one

        Args:
            user_id: User identifier
            chapter_context: Current chapter

        Returns:
            Conversation ID
        """
        # Try to find recent conversation in same chapter
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                if chapter_context:
                    cur.execute("""
                        SELECT id FROM conversations
                        WHERE user_id = %s AND chapter_context = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """, (user_id, chapter_context))
                else:
                    cur.execute("""
                        SELECT id FROM conversations
                        WHERE user_id = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """, (user_id,))

                result = cur.fetchone()

                if result:
                    return result['id']
                else:
                    # Create new conversation
                    return self.create_conversation(user_id, chapter_context)

    def delete_conversation(self, conversation_id: int):
        """Delete a conversation and all its messages"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM conversations WHERE id = %s
                """, (conversation_id,))

# Singleton instance
_neon_service = None

def get_neon_service() -> NeonService:
    """Get or create Neon service singleton"""
    global _neon_service
    if _neon_service is None:
        _neon_service = NeonService()
    return _neon_service
