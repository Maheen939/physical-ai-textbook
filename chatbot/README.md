# Physical AI Textbook - RAG Chatbot

Complete RAG (Retrieval-Augmented Generation) chatbot implementation for the Physical AI & Humanoid Robotics textbook.

## Features

✅ **Context-Aware Responses**: Chatbot answers based on textbook content
✅ **Semantic Search**: Qdrant vector database for finding relevant content
✅ **Conversation History**: Persistent chat history with Neon Postgres
✅ **Chapter-Specific Help**: Filter responses by current chapter
✅ **Selected Text Queries**: Ask questions about highlighted text
✅ **OpenAI GPT-4**: High-quality, educational responses

## Architecture

```
User Query
    ↓
Generate Embedding (OpenAI)
    ↓
Search Similar Content (Qdrant)
    ↓
Retrieve Context + History (Neon Postgres)
    ↓
Generate Response (GPT-4 + RAG)
    ↓
Store Conversation (Neon Postgres)
    ↓
Return Response to User
```

## Setup

### 1. Prerequisites

- Python 3.10+
- OpenAI API key
- Qdrant Cloud account (free tier)
- Neon Serverless Postgres account (free tier)

### 2. Get API Keys

**OpenAI**:
1. Go to https://platform.openai.com/api-keys
2. Create new API key
3. Copy key (starts with `sk-`)

**Qdrant Cloud**:
1. Sign up at https://cloud.qdrant.io/
2. Create a cluster (free tier: 1GB)
3. Copy cluster URL and API key

**Neon Postgres**:
1. Sign up at https://neon.tech/
2. Create a project (free tier: 0.5GB)
3. Copy connection string

### 3. Install Dependencies

```bash
cd chatbot
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp ../.env.example .env
# Edit .env with your API keys
```

`.env` file:
```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key

# Neon Postgres
DATABASE_URL=postgres://user:password@host/database

# Configuration
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4
MAX_CONTEXT_LENGTH=8000
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
```

### 5. Initialize Services

```bash
python -c "from api.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)" &

# In another terminal, initialize:
curl -X POST http://localhost:8000/api/init \
  -H "Content-Type: application/json" \
  -d '{"reset": false}'
```

Or use Python:
```python
import requests

# Initialize database and vector store
response = requests.post("http://localhost:8000/api/init", json={"reset": False})
print(response.json())
```

### 6. Generate Embeddings

```bash
python scripts/generate_embeddings.py
```

This will:
- Read all markdown files from `../docs/`
- Split content into chunks (500 words, 50 overlap)
- Generate embeddings using OpenAI
- Upload vectors to Qdrant

**Note**: This may take 10-15 minutes and cost ~$1-2 in OpenAI API usage.

## Usage

### Start the API Server

```bash
python api/main.py
```

API runs at: http://localhost:8000

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Initialize Services
```bash
POST /api/init
Body: {"reset": false}
```

#### Chat Query
```bash
POST /api/chat/query
Body: {
  "user_id": "user123",
  "message": "What is ROS 2?",
  "chapter_context": "module-1/ros2-fundamentals",
  "selected_text": null
}
```

Response:
```json
{
  "response": "ROS 2 (Robot Operating System 2) is...",
  "sources": ["Module 1: ROS 2 Fundamentals - Introduction"],
  "conversation_id": 1
}
```

#### Get User Chat History
```bash
GET /api/chat/history/{user_id}
```

#### Get Conversation Messages
```bash
GET /api/chat/conversation/{conversation_id}
```

## Testing

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Chat query
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "What are ROS 2 nodes?",
    "chapter_context": "module-1/ros2-fundamentals"
  }'
```

### Test with Python

```python
import requests

# Chat query
response = requests.post(
    "http://localhost:8000/api/chat/query",
    json={
        "user_id": "test_user",
        "message": "Explain ROS 2 topics",
        "chapter_context": "module-1/ros2-fundamentals"
    }
)

result = response.json()
print("Response:", result["response"])
print("Sources:", result["sources"])
print("Conversation ID:", result["conversation_id"])
```

## Project Structure

```
chatbot/
├── api/
│   ├── main.py                    # FastAPI application
│   └── services/
│       ├── openai_service.py      # OpenAI embeddings + chat
│       ├── qdrant_service.py      # Vector database
│       └── neon_service.py        # Postgres conversation storage
├── scripts/
│   └── generate_embeddings.py     # Embedding generation script
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (create this)
└── README.md                     # This file
```

## Cost Estimation

### One-Time Setup
- **Embedding Generation**: ~$1-2 (one time)
  - 12 chapters × ~4,000 words = 48,000 words
  - ~36,000 tokens × $0.00002 per token = $0.72

### Per Query
- **User Query Embedding**: $0.00002 per query
- **GPT-4 Chat Completion**: ~$0.01-0.03 per query
  - Input: ~1,000 tokens (context + query) × $0.00003 = $0.03
  - Output: ~300 tokens × $0.00006 = $0.018

**Total per query**: ~$0.05

**For 100 queries**: ~$5
**For 1,000 queries**: ~$50

### Optimization Tips
1. **Cache responses** for common queries
2. **Reduce context** by using chapter filtering
3. **Use GPT-3.5-turbo** for cheaper responses (lower quality)
4. **Batch embeddings** when generating

## Deployment

### Deploy to Vercel (Recommended)

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy:
```bash
cd chatbot
vercel deploy
```

3. Add environment variables in Vercel dashboard:
   - `OPENAI_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `DATABASE_URL`

### Deploy to Railway

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Deploy:
```bash
cd chatbot
railway up
```

3. Add environment variables in Railway dashboard

## Troubleshooting

### "OPENAI_API_KEY not set"
- Make sure `.env` file exists in `chatbot/` directory
- Check that `python-dotenv` is installed
- Verify API key is correct (starts with `sk-`)

### "Cannot connect to Qdrant"
- Verify cluster URL (should include `https://`)
- Check API key is correct
- Ensure cluster is running (check Qdrant dashboard)

### "Database connection failed"
- Verify Neon connection string format
- Check database exists and is accessible
- Ensure IP is whitelisted (Neon auto-whitelists)

### "No results from vector search"
- Run `generate_embeddings.py` to populate Qdrant
- Check collection exists: `GET /` endpoint shows collection info
- Verify embeddings were uploaded successfully

### "Out of memory"
- Reduce `CHUNK_SIZE` in `.env`
- Process fewer files at once in `generate_embeddings.py`
- Use batch_size parameter in embedding generation

## API Documentation

Auto-generated docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Next Steps

1. ✅ Backend API complete
2. ⏳ Create React chatbot widget component
3. ⏳ Integrate widget into Docusaurus
4. ⏳ Deploy chatbot backend
5. ⏳ Deploy frontend with chatbot

## Support

For issues or questions:
- Check troubleshooting section above
- Review API docs at `/docs` endpoint
- Check service status at `/health` endpoint

---

**Built for the Physical AI & Humanoid Robotics Textbook Hackathon**
