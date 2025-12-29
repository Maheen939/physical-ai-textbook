"""
Physical AI Textbook RAG Chatbot API
FastAPI backend for context-aware chatbot with Qdrant and Neon Postgres
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Import services
from api.services.qdrant_service import get_qdrant_service, QdrantService
from api.services.neon_service import get_neon_service, NeonService
from api.auth_routes import router as auth_router
from api.personalization_routes import router as personalization_router
from api.translation_routes import router as translation_router

# Conditional AI service import based on environment
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()

if AI_PROVIDER == "openai":
    from api.services.openai_service import get_openai_service as get_ai_service, OpenAIService as AIService
else:
    from api.services.gemini_service import get_gemini_service as get_ai_service, GeminiService as AIService

load_dotenv()

app = FastAPI(
    title="Physical AI Textbook Chatbot API",
    description="RAG chatbot for interactive learning",
    version="1.0.0"
)

# Include authentication routes
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])

# Include personalization routes
app.include_router(personalization_router, prefix="/api/personalize", tags=["Personalization"])

# Include translation routes
app.include_router(translation_router, prefix="/api/translate", tags=["Translation"])

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        os.getenv("FRONTEND_URL", "http://localhost:3000")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatQuery(BaseModel):
    user_id: Optional[str] = "anonymous"
    message: str
    chapter_context: Optional[str] = None
    selected_text: Optional[str] = None
    conversation_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    conversation_id: int

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict

class InitRequest(BaseModel):
    reset: bool = False

# Routes
@app.get("/", response_model=HealthResponse)
async def root(
    ai_svc: AIService = Depends(get_ai_service),
    qdrant_svc: QdrantService = Depends(get_qdrant_service),
    neon_svc: NeonService = Depends(get_neon_service)
):
    """Health check endpoint"""
    try:
        qdrant_info = qdrant_svc.get_collection_info()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            services={
                "ai_provider": AI_PROVIDER,
                "qdrant": qdrant_info,
                "neon": "connected"
            }
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            version="1.0.0",
            services={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy"}

@app.post("/api/init")
async def initialize_services(
    request: InitRequest,
    qdrant_svc: QdrantService = Depends(get_qdrant_service),
    neon_svc: NeonService = Depends(get_neon_service)
):
    """Initialize database and vector store"""
    try:
        if request.reset:
            # Reset Qdrant collection
            try:
                qdrant_svc.delete_collection()
            except:
                pass

        # Create Qdrant collection
        qdrant_svc.create_collection()

        # Initialize Neon database
        neon_svc.init_database()

        return {
            "status": "success",
            "message": "Services initialized successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/query", response_model=ChatResponse)
async def chat_query(
    query: ChatQuery,
    ai_svc: AIService = Depends(get_ai_service),
    qdrant_svc: QdrantService = Depends(get_qdrant_service),
    neon_svc: NeonService = Depends(get_neon_service)
):
    """
    Process chat query with RAG (Retrieval-Augmented Generation)

    Args:
        query: ChatQuery with user_id, message, chapter_context, selected_text

    Returns:
        ChatResponse with response, sources, conversation_id
    """
    try:
        # Get or create conversation
        if query.conversation_id:
            conv_id = query.conversation_id
        else:
            conv_id = neon_svc.get_or_create_conversation(
                query.user_id,
                query.chapter_context
            )

        # Generate embedding for user query
        query_embedding = ai_svc.generate_embedding(query.message)

        # Search for relevant context
        top_k = int(os.getenv("TOP_K_RESULTS", "5"))
        retrieved_contexts = qdrant_svc.search(
            query_vector=query_embedding,
            limit=top_k,
            chapter_filter=query.chapter_context if query.chapter_context else None,
            score_threshold=0.7
        )

        # Get conversation history for context
        conversation_history = neon_svc.get_conversation_history(conv_id, limit=10)
        history_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation_history[-6:]  # Last 3 exchanges
        ]

        # Generate response with RAG
        response = ai_svc.chat_with_rag(
            user_query=query.message,
            retrieved_contexts=retrieved_contexts,
            chapter_context=query.chapter_context,
            selected_text=query.selected_text,
            conversation_history=history_messages if history_messages else None
        )

        # Store user message and assistant response
        neon_svc.add_message(conv_id, "user", query.message)
        neon_svc.add_message(conv_id, "assistant", response)

        # Extract source information
        sources = [
            f"{ctx['chapter']}" + (f" - {ctx['section']}" if ctx['section'] else "")
            for ctx in retrieved_contexts
        ]

        return ChatResponse(
            response=response,
            sources=sources,
            conversation_id=conv_id
        )

    except Exception as e:
        print(f"Error in chat_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{user_id}")
async def get_chat_history(
    user_id: str,
    limit: int = 20,
    neon_svc: NeonService = Depends(get_neon_service)
):
    """
    Get chat history for a user

    Args:
        user_id: User identifier
        limit: Maximum number of conversations to return

    Returns:
        List of conversations with metadata
    """
    try:
        conversations = neon_svc.get_user_conversations(user_id, limit)
        return {
            "user_id": user_id,
            "conversations": conversations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/conversation/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    neon_svc: NeonService = Depends(get_neon_service)
):
    """Get messages for a specific conversation"""
    try:
        messages = neon_svc.get_conversation_history(conversation_id, limit=100)
        return {
            "conversation_id": conversation_id,
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
