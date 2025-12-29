# Chatbot Setup Guide

Complete guide to set up and use the RAG chatbot in the Physical AI textbook.

## Overview

The chatbot is a floating AI tutor that helps students understand course content. It uses:
- **RAG (Retrieval-Augmented Generation)** for context-aware responses
- **OpenAI GPT-4** for natural language understanding
- **Qdrant** for semantic search through textbook content
- **Neon Postgres** for conversation history

## Quick Start

### 1. Backend Setup

```bash
cd chatbot

# Install dependencies
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp ../.env.example .env
# Edit .env with your API keys (see below)

# Initialize services
python api/main.py &
curl -X POST http://localhost:8000/api/init -H "Content-Type: application/json" -d '{"reset": false}'

# Generate embeddings (takes 10-15 minutes)
python scripts/generate_embeddings.py
```

### 2. Frontend Setup

```bash
# Back to project root
cd ..

# Configure API URL
cp .env.local.example .env.local
# Edit REACT_APP_API_URL if needed (default: http://localhost:8000)

# Start development server
npm start
```

The chatbot will appear as a floating purple button in the bottom-right corner! 💬

## Required API Keys

### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy key (starts with `sk-`)
4. Add to `.env`: `OPENAI_API_KEY=sk-...`

**Cost**: ~$0.05 per chat query

### Qdrant Cloud
1. Sign up at https://cloud.qdrant.io/
2. Create cluster (Free tier: 1GB, no credit card)
3. Copy cluster URL and API key
4. Add to `.env`:
   ```
   QDRANT_URL=https://xyz.qdrant.io
   QDRANT_API_KEY=...
   ```

### Neon Postgres
1. Sign up at https://neon.tech/
2. Create project (Free tier: 0.5GB, no credit card)
3. Copy connection string
4. Add to `.env`: `DATABASE_URL=postgres://...`

## Features

### 1. **Ask Questions**
Type any question about the course:
- "What is ROS 2?"
- "How do I create a URDF file?"
- "Explain SLAM algorithms"

### 2. **Chapter Context**
Chatbot automatically knows which chapter you're viewing and provides relevant answers.

### 3. **Select Text to Ask**
1. Highlight any text on the page
2. Click the "🤔 Ask AI" button that appears
3. Chatbot will explain the selected text

### 4. **View Sources**
Each response shows which chapters the information came from. Click "📚 Sources" to see them.

### 5. **Conversation History**
Chatbot remembers previous messages in the conversation for context.

## Component Files

```
src/
├── components/
│   ├── ChatbotWidget.tsx          # Main chatbot UI
│   └── ChatbotWidget.module.css   # Styles
├── hooks/
│   └── useChatbot.ts              # API communication logic
└── theme/
    └── Root.tsx                   # Docusaurus integration
```

## Customization

### Change Chatbot Position

Edit `ChatbotWidget.module.css`:
```css
.chatButton {
  bottom: 24px;  /* Change this */
  right: 24px;   /* Or this */
}
```

### Change Colors

Edit `ChatbotWidget.module.css`:
```css
.chatButton {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  /* Change gradient colors here */
}
```

### Modify Welcome Message

Edit `ChatbotWidget.tsx`:
```tsx
<div className={styles.welcomeMessage}>
  <h4>👋 Welcome!</h4>
  {/* Edit welcome text here */}
</div>
```

### Change AI Model

Edit `chatbot/.env`:
```bash
CHAT_MODEL=gpt-3.5-turbo  # Cheaper, faster, lower quality
# or
CHAT_MODEL=gpt-4          # More expensive, slower, higher quality
```

## Testing

### Test in Browser

1. Start backend: `cd chatbot && python api/main.py`
2. Start frontend: `npm start`
3. Open http://localhost:3000
4. Click chatbot button
5. Ask: "What is ROS 2?"

### Test Backend Directly

```bash
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "message": "What is ROS 2?",
    "chapter_context": "module-1/ros2-fundamentals"
  }'
```

## Deployment

### Backend Deployment

**Option 1: Vercel**
```bash
cd chatbot
vercel deploy
```

**Option 2: Railway**
```bash
cd chatbot
railway up
```

Add environment variables in deployment dashboard:
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `DATABASE_URL`

### Frontend Deployment

Update `.env.local`:
```bash
REACT_APP_API_URL=https://your-chatbot-api.vercel.app
```

Deploy to GitHub Pages:
```bash
npm run build
npm run deploy
```

## Troubleshooting

### Chatbot button not showing
- Check browser console for errors
- Verify `Root.tsx` is in `src/theme/` directory
- Clear browser cache and reload

### "Failed to fetch" error
- Ensure backend is running: `curl http://localhost:8000/health`
- Check `REACT_APP_API_URL` in `.env.local`
- Check CORS settings in `chatbot/api/main.py`

### Slow responses
- Normal: First query after restart takes 2-3 seconds
- Check OpenAI API status
- Reduce `TOP_K_RESULTS` in backend `.env` (default: 5)

### No relevant answers
- Run `generate_embeddings.py` to populate Qdrant
- Check Qdrant has vectors: `curl http://localhost:8000/`
- Verify chapter context is correct

### API costs too high
- Switch to `gpt-3.5-turbo` (10x cheaper)
- Reduce `TOP_K_RESULTS` to 3
- Implement response caching
- Add rate limiting

## Cost Management

### Typical Costs

**Setup (one-time)**:
- Embedding generation: ~$1-2

**Per 1,000 queries**:
- With GPT-4: ~$50
- With GPT-3.5-turbo: ~$5

### Cost Optimization

1. **Use GPT-3.5-turbo** for lower costs:
   ```bash
   CHAT_MODEL=gpt-3.5-turbo
   ```

2. **Reduce context**:
   ```bash
   TOP_K_RESULTS=3  # Instead of 5
   ```

3. **Cache responses** (implement in `useChatbot.ts`):
   ```typescript
   const cache = new Map();
   if (cache.has(message)) return cache.get(message);
   ```

4. **Rate limiting** (implement in backend):
   ```python
   # Limit to 10 queries per user per hour
   ```

## Advanced Features

### Add Voice Input

Install Web Speech API:
```tsx
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();

recognition.onresult = (event) => {
  const transcript = event.results[0][0].transcript;
  setInputValue(transcript);
};

<button onClick={() => recognition.start()}>🎤</button>
```

### Add Streaming Responses

Modify `useChatbot.ts` to use streaming:
```typescript
const response = await fetch(`${API_BASE_URL}/api/chat/query`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ...data, stream: true })
});

const reader = response.body.getReader();
// Handle streaming chunks...
```

### Add User Analytics

Track usage:
```typescript
import { analytics } from './analytics';

const sendMessage = async (message) => {
  analytics.track('chatbot_query', {
    message_length: message.length,
    chapter: chapterContext,
  });
  // ... rest of code
};
```

## Support

- **Backend Issues**: Check `chatbot/README.md`
- **API Documentation**: http://localhost:8000/docs
- **Frontend Issues**: Check browser console

## Next Steps

1. ✅ Backend API running
2. ✅ Frontend chatbot integrated
3. ⏳ Deploy backend to Vercel/Railway
4. ⏳ Update frontend with production API URL
5. ⏳ Deploy frontend to GitHub Pages
6. ⏳ Test end-to-end in production

---

**Happy chatting! 🤖💬**
