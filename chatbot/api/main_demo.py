"""
Demo FastAPI App - For testing chatbot without API keys
Uses built-in knowledge base instead of external services
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

# Import demo service
from api.demo_service import router as demo_router, find_relevant_content

app = FastAPI(
    title="Physical AI Textbook Chatbot API (Demo)",
    description="Demo chatbot for testing without API keys",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include demo routes
app.include_router(demo_router, prefix="/api")

# Simple chat model
class ChatQuery(BaseModel):
    user_id: Optional[str] = "anonymous"
    message: str
    chapter_context: Optional[str] = None
    selected_text: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    conversation_id: int

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "mode": "demo",
        "message": "Demo mode - no API keys required"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/chat/query", response_model=ChatResponse)
async def chat_query(query: ChatQuery):
    """Demo chat that uses knowledge base instead of RAG."""
    relevant, sources = find_relevant_content(query.message)

    response = f"""# Physical AI Learning Assistant

Based on your question about: **{query.message}**

{relevant}

---

**Tips:**
- Ask follow-up questions for more details
- Ask for code examples
- Request clarification on any concept

Is there anything else I can help you with?"""

    return ChatResponse(
        response=response,
        sources=sources,
        conversation_id=1
    )

@app.get("/api/chat/history/{user_id}")
async def get_chat_history(user_id: str):
    return {"user_id": user_id, "conversations": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
