"""
OpenAI Service for embeddings and chat completion
"""

import os
from typing import List, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAIService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4")
        self.max_tokens = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using OpenAI API

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """
        Generate chat completion using GPT-4

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response

        Returns:
            Chat completion response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 1000,
                stream=stream
            )

            if stream:
                return response
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error in chat completion: {e}")
            raise

    def build_rag_prompt(
        self,
        user_query: str,
        retrieved_contexts: List[dict],
        chapter_context: Optional[str] = None,
        selected_text: Optional[str] = None
    ) -> List[dict]:
        """
        Build RAG prompt with retrieved context

        Args:
            user_query: User's question
            retrieved_contexts: List of retrieved context dicts with 'content', 'chapter', 'section'
            chapter_context: Current chapter being viewed (optional)
            selected_text: Text selected by user (optional)

        Returns:
            List of messages for chat completion
        """
        # Build context string
        context_parts = []

        if selected_text:
            context_parts.append(f"**Selected Text**:\n{selected_text}\n")

        if chapter_context:
            context_parts.append(f"**Current Chapter**: {chapter_context}\n")

        if retrieved_contexts:
            context_parts.append("**Relevant Content**:")
            for i, ctx in enumerate(retrieved_contexts, 1):
                chapter = ctx.get('chapter', 'Unknown')
                section = ctx.get('section', '')
                content = ctx.get('content', '')
                context_parts.append(
                    f"\n[{i}] From {chapter}"
                    + (f" - {section}" if section else "")
                    + f":\n{content}\n"
                )

        context_str = "\n".join(context_parts)

        # System prompt
        system_prompt = """You are an expert AI tutor for the Physical AI & Humanoid Robotics course.
Your role is to help students understand robotics concepts, ROS 2, simulation, NVIDIA Isaac, and vision-language-action systems.

Guidelines:
1. Provide clear, educational explanations suitable for students
2. Reference specific chapters and sections when relevant
3. Use code examples when helpful
4. Break down complex concepts step-by-step
5. Encourage hands-on learning and experimentation
6. If you don't know something, say so honestly
7. Relate concepts to practical robotics applications

When answering:
- Start with a direct answer
- Provide additional context and examples
- Suggest related topics to explore
- Reference relevant chapters for deeper learning
"""

        # User prompt with context
        user_prompt = f"""Context from the textbook:
{context_str}

Student Question: {user_query}

Please provide a helpful, educational response based on the context above."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return messages

    def chat_with_rag(
        self,
        user_query: str,
        retrieved_contexts: List[dict],
        chapter_context: Optional[str] = None,
        selected_text: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        """
        Generate RAG-enhanced chat response

        Args:
            user_query: User's question
            retrieved_contexts: Retrieved context from vector DB
            chapter_context: Current chapter (optional)
            selected_text: Selected text (optional)
            conversation_history: Previous conversation (optional)

        Returns:
            Generated response
        """
        messages = self.build_rag_prompt(
            user_query,
            retrieved_contexts,
            chapter_context,
            selected_text
        )

        # Add conversation history if provided
        if conversation_history:
            # Insert history before the current query
            messages = [messages[0]] + conversation_history + [messages[1]]

        response = self.chat_completion(messages, temperature=0.7)
        return response

# Singleton instance
_openai_service = None

def get_openai_service() -> OpenAIService:
    """Get or create OpenAI service singleton"""
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service
