"""
Google Gemini Service for embeddings and chat
100% FREE alternative to OpenAI
"""

import os
import google.generativeai as genai
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

class GeminiService:
    def __init__(self):
        """Initialize Gemini with API key"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        # Models
        self.embedding_model = "models/embedding-001"
        self.chat_model = genai.GenerativeModel('gemini-pro')

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            List of floats (768 dimensions for Gemini)
        """
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Gemini can handle batch requests
        for i in range(0, len(texts), 100):  # Process in batches of 100
            batch = texts[i:i+100]

            try:
                # Generate embeddings for batch
                for text in batch:
                    embedding = self.generate_embedding(text)
                    embeddings.append(embedding)

                print(f"Generated embeddings for batch {i//100 + 1}")

            except Exception as e:
                print(f"Error in batch {i//100 + 1}: {e}")
                # Continue with next batch
                continue

        return embeddings

    def chat_with_rag(
        self,
        user_query: str,
        retrieved_contexts: List[Dict],
        chapter_context: Optional[str] = None,
        selected_text: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate chat response with RAG

        Args:
            user_query: User's question
            retrieved_contexts: Relevant contexts from vector search
            chapter_context: Current chapter user is reading
            selected_text: Text user highlighted (optional)
            conversation_history: Previous messages (optional)

        Returns:
            AI-generated response
        """
        # Build context from retrieved documents
        context_text = ""
        for i, ctx in enumerate(retrieved_contexts, 1):
            context_text += f"\n--- Context {i} (from {ctx['chapter']}) ---\n"
            context_text += ctx['content']
            context_text += "\n"

        # Build the prompt
        prompt = self._build_rag_prompt(
            user_query=user_query,
            context_text=context_text,
            chapter_context=chapter_context,
            selected_text=selected_text,
            conversation_history=conversation_history
        )

        # Generate response
        try:
            response = self.chat_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating chat response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    def _build_rag_prompt(
        self,
        user_query: str,
        context_text: str,
        chapter_context: Optional[str] = None,
        selected_text: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Build the prompt for RAG"""

        system_instructions = """You are an expert AI tutor for Physical AI and Humanoid Robotics.
You help students understand complex robotics concepts including ROS 2, Gazebo simulation,
NVIDIA Isaac, and Vision-Language-Action models.

Guidelines:
1. Provide clear, accurate, and educational responses
2. Use the provided context from the textbook
3. Include code examples when relevant
4. Break down complex concepts into understandable parts
5. Be encouraging and supportive of student learning
6. If information isn't in the context, say so honestly
7. Reference specific chapters when citing information"""

        prompt = system_instructions + "\n\n"

        # Add chapter context
        if chapter_context:
            prompt += f"\nCurrent Chapter: {chapter_context}\n"

        # Add selected text context
        if selected_text:
            prompt += f"\nStudent highlighted this text: \"{selected_text}\"\n"

        # Add conversation history
        if conversation_history:
            prompt += "\nConversation History:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt += f"{role.capitalize()}: {content}\n"

        # Add retrieved context
        prompt += f"\nRelevant Textbook Content:\n{context_text}\n"

        # Add user query
        prompt += f"\nStudent Question: {user_query}\n"
        prompt += "\nYour Response:"

        return prompt

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Simple chat completion (non-RAG)

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Max response length

        Returns:
            AI response text
        """
        # Convert messages to prompt
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt += f"{role.capitalize()}: {content}\n"

        prompt += "Assistant:"

        try:
            response = self.chat_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return "I apologize, but I encountered an error. Please try again."


# Singleton instance
_gemini_service = None

def get_gemini_service() -> GeminiService:
    """Get or create Gemini service singleton"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
