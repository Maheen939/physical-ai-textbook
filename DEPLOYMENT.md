# Production Deployment Guide

Complete guide to deploy the Physical AI Textbook with RAG Chatbot to production.

## Overview

**Architecture**:
- **Frontend**: GitHub Pages (Docusaurus static site)
- **Backend (Chatbot API)**: Railway or Vercel (your choice)
- **Databases**: Qdrant Cloud (free) + Neon Postgres (free)

**Total Cost**: $0-5/month (depending on usage)

---

## Part 1: Get Free API Keys (Required)

### 1.1 OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up / Login
3. Click "Create new secret key"
4. Copy key (starts with `sk-`)
5. **Add $5-10 credit** to your account

**Cost**: ~$0.05 per chatbot query

### 1.2 Qdrant Cloud (Vector Database)

1. Go to https://cloud.qdrant.io/
2. Sign up (no credit card required)
3. Click "Create Cluster"
   - Name: `physical-ai-textbook`
   - Region: Choose closest to you
   - Plan: **Free** (1GB, perfect for this project)
4. Wait for cluster to be created (~2 minutes)
5. Copy:
   - **Cluster URL**: `https://xyz-abc.qdrant.io`
   - **API Key**: Click "API Keys" → Copy key

**Cost**: FREE (1GB storage, 1M vectors)

### 1.3 Neon Postgres (Conversation Storage)

1. Go to https://neon.tech/
2. Sign up (no credit card required)
3. Click "Create Project"
   - Name: `physical-ai-chatbot`
   - Region: Choose closest to you
   - Plan: **Free** (0.5GB, perfect for this project)
4. Copy **Connection String**:
   - Format: `postgres://user:pass@host/database`

**Cost**: FREE (0.5GB storage)

---

## Part 2: Deploy Chatbot Backend

### Option A: Railway (Recommended - Easier)

#### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
```

Or use web interface: https://railway.app/

#### Step 2: Deploy Backend

```bash
cd physical-ai-textbook/chatbot

# Login to Railway
railway login

# Initialize project
railway init

# Add environment variables
railway variables set OPENAI_API_KEY="sk-your-key-here"
railway variables set QDRANT_URL="https://your-cluster.qdrant.io"
railway variables set QDRANT_API_KEY="your-qdrant-key"
railway variables set DATABASE_URL="postgres://user:pass@host/database"
railway variables set EMBEDDING_MODEL="text-embedding-3-small"
railway variables set CHAT_MODEL="gpt-4"
railway variables set TOP_K_RESULTS="5"

# Deploy
railway up
```

#### Step 3: Get Backend URL

```bash
railway domain
```

Copy the URL (e.g., `https://your-app.railway.app`)

#### Step 4: Initialize Services

```bash
curl -X POST https://your-app.railway.app/api/init \
  -H "Content-Type: application/json" \
  -d '{"reset": false}'
```

### Option B: Vercel (Alternative)

#### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

#### Step 2: Deploy Backend

```bash
cd physical-ai-textbook/chatbot

# Login
vercel login

# Deploy
vercel

# Add environment variables via Vercel dashboard:
# https://vercel.com/your-username/your-project/settings/environment-variables

# Redeploy after adding variables
vercel --prod
```

---

## Part 3: Generate Embeddings

**Important**: Do this AFTER backend is deployed and initialized.

### Local Method (Recommended)

```bash
cd physical-ai-textbook/chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
DATABASE_URL=postgres://user:pass@host/database
EOF

# Generate embeddings (takes 10-15 minutes)
python scripts/generate_embeddings.py
```

**Expected Output**:
```
Found 20 markdown files
Processing: intro.md
  - Created 5 chunks
Processing: prerequisites.md
  - Created 8 chunks
...
Total chunks: 450
Generating embeddings for batch 1...
Generating embeddings for batch 2...
...
Uploading 450 vectors to Qdrant...
✅ Successfully uploaded all vectors!

Collection Info:
  - Vectors: 450
  - Points: 450
```

**Cost**: ~$1-2 one-time

### Verify Embeddings

```bash
curl https://your-app.railway.app/

# Should show:
# {
#   "status": "healthy",
#   "services": {
#     "qdrant": {
#       "vectors_count": 450,
#       "points_count": 450
#     }
#   }
# }
```

---

## Part 4: Deploy Frontend (GitHub Pages)

### Step 1: Update Docusaurus Config

Edit `docusaurus.config.ts`:

```typescript
const config: Config = {
  url: 'https://YOUR-USERNAME.github.io',
  baseUrl: '/physical-ai-textbook/',
  organizationName: 'YOUR-USERNAME',
  projectName: 'physical-ai-textbook',
  // ... rest of config
};
```

### Step 2: Configure Chatbot API URL

Create `.env.local`:

```bash
cd physical-ai-textbook
cat > .env.local << EOF
REACT_APP_API_URL=https://your-app.railway.app
EOF
```

### Step 3: Deploy to GitHub Pages

```bash
# Build and deploy
npm run build
npm run deploy
```

Or configure GitHub Actions (auto-deploys on push):

Create `.github/workflows/deploy.yml` - already exists!

### Step 4: Enable GitHub Pages

1. Go to your GitHub repo
2. Settings → Pages
3. Source: **gh-pages branch**
4. Click Save

Wait 2-3 minutes, then visit:
`https://YOUR-USERNAME.github.io/physical-ai-textbook/`

---

## Part 5: Test Production Deployment

### Test Backend

```bash
# Health check
curl https://your-app.railway.app/health

# Test chat query
curl -X POST https://your-app.railway.app/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "message": "What is ROS 2?",
    "chapter_context": "module-1/ros2-fundamentals"
  }'
```

Expected response (within 2-3 seconds):
```json
{
  "response": "ROS 2 (Robot Operating System 2) is...",
  "sources": ["Module 1: ROS 2 Fundamentals"],
  "conversation_id": 1
}
```

### Test Frontend

1. Open: `https://YOUR-USERNAME.github.io/physical-ai-textbook/`
2. Click purple chatbot button (bottom-right)
3. Ask: "What is ROS 2?"
4. Should receive response within 3 seconds

### Test Features

✅ **Chapter Context**:
- Navigate to any chapter
- Ask a question
- Bot should reference current chapter

✅ **Selected Text**:
- Highlight any text on the page
- Click "Ask AI" button
- Bot explains selected text

✅ **Conversation History**:
- Ask multiple questions
- Bot remembers context

✅ **Sources**:
- Click "📚 Sources" in bot response
- Shows which chapters used

---

## Part 6: Monitor and Maintain

### Check Backend Logs

**Railway**:
```bash
railway logs
```

Or visit: https://railway.app/dashboard → Your Project → Deployments

**Vercel**:
Visit: https://vercel.com/dashboard → Your Project → Logs

### Monitor Costs

**OpenAI**:
- Visit: https://platform.openai.com/usage
- Check daily usage

**Expected Costs**:
- 100 queries: ~$5
- 1,000 queries: ~$50

### Monitor Database

**Qdrant**:
- Visit: https://cloud.qdrant.io/
- Check vector count (should be ~450)

**Neon**:
- Visit: https://console.neon.tech/
- Check storage usage

---

## Troubleshooting

### Backend Issues

#### "Service Unhealthy"
```bash
# Check logs
railway logs

# Common issues:
# 1. Missing environment variables
# 2. Invalid API keys
# 3. Database connection failed
```

#### "No vectors found"
```bash
# Re-run embedding generation
python scripts/generate_embeddings.py

# Check Qdrant
curl https://your-app.railway.app/
```

#### "OpenAI API error"
```bash
# Check API key is valid
# Check you have credits: https://platform.openai.com/account/billing
```

### Frontend Issues

#### "Chatbot button not showing"
```bash
# Check browser console (F12)
# Verify .env.local has correct API URL
# Clear browser cache
```

#### "Failed to fetch"
```bash
# Check backend is running
curl https://your-app.railway.app/health

# Check CORS settings in api/main.py
# Verify REACT_APP_API_URL in .env.local
```

#### "Slow responses"
```bash
# Normal: 2-3 seconds
# If > 5 seconds:
# 1. Check OpenAI API status
# 2. Reduce TOP_K_RESULTS
# 3. Switch to gpt-3.5-turbo
```

---

## Cost Optimization

### Use GPT-3.5-Turbo (10x cheaper)

Update Railway variables:
```bash
railway variables set CHAT_MODEL="gpt-3.5-turbo"
```

**Trade-off**: Lower quality responses, but 90% cheaper

### Reduce Context

```bash
railway variables set TOP_K_RESULTS="3"
```

**Trade-off**: Less context, but faster and cheaper

### Implement Caching

Edit `src/hooks/useChatbot.ts` to cache responses locally.

---

## Security Checklist

✅ **Never commit API keys** to Git
✅ **Environment variables** set in Railway/Vercel
✅ **CORS configured** in `api/main.py`
✅ **Rate limiting** (TODO: implement if needed)
✅ **Input validation** in chatbot hook

---

## Maintenance

### Update Content

When you add new chapters:

1. **Add markdown file** to `docs/`
2. **Regenerate embeddings**:
   ```bash
   python scripts/generate_embeddings.py
   ```
3. **Redeploy frontend**:
   ```bash
   npm run deploy
   ```

### Update Backend

```bash
cd chatbot
# Make changes
railway up  # or vercel --prod
```

### Update Frontend

```bash
npm run deploy
```

Or push to GitHub (auto-deploys with Actions)

---

## Quick Reference

### Important URLs

- **Frontend**: `https://YOUR-USERNAME.github.io/physical-ai-textbook/`
- **Backend API**: `https://your-app.railway.app`
- **API Docs**: `https://your-app.railway.app/docs`
- **Health Check**: `https://your-app.railway.app/health`

### Environment Variables

```bash
# Backend (.env or Railway variables)
OPENAI_API_KEY=sk-...
QDRANT_URL=https://...qdrant.io
QDRANT_API_KEY=...
DATABASE_URL=postgres://...
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4
TOP_K_RESULTS=5

# Frontend (.env.local)
REACT_APP_API_URL=https://your-app.railway.app
```

### Common Commands

```bash
# Deploy backend
cd chatbot && railway up

# Deploy frontend
npm run deploy

# View logs
railway logs

# Generate embeddings
cd chatbot && python scripts/generate_embeddings.py

# Test backend
curl https://your-app.railway.app/health
```

---

## Success Checklist

Before submitting to hackathon:

### Backend
- ✅ Deployed to Railway/Vercel
- ✅ All environment variables set
- ✅ Health check returns "healthy"
- ✅ Qdrant has ~450 vectors
- ✅ Database initialized
- ✅ Chat endpoint responds < 3s

### Frontend
- ✅ Deployed to GitHub Pages
- ✅ All pages load correctly
- ✅ Chatbot button visible
- ✅ Chat window opens/closes
- ✅ Messages send and receive
- ✅ Sources display correctly
- ✅ Mobile responsive

### Testing
- ✅ Ask 5 different questions
- ✅ Test chapter context
- ✅ Test selected text
- ✅ Test conversation history
- ✅ Test on mobile device

### Documentation
- ✅ README updated with live URLs
- ✅ Demo video created (< 90 seconds)
- ✅ GitHub repo public

---

## Demo Video Script (90 seconds)

**0:00-0:15** - Show homepage and navigation
**0:15-0:30** - Click chatbot, ask "What is ROS 2?"
**0:30-0:45** - Navigate to chapter, ask chapter-specific question
**0:45-0:60** - Highlight text, click "Ask AI"
**0:60-0:75** - Show sources, conversation history
**0:75-0:90** - Show mobile responsive design

---

## Support

- **Backend Issues**: Check `chatbot/README.md`
- **Frontend Issues**: Check `CHATBOT_SETUP.md`
- **API Docs**: https://your-app.railway.app/docs
- **Railway Docs**: https://docs.railway.app/
- **Vercel Docs**: https://vercel.com/docs

---

## Next Steps

1. ✅ Get API keys (OpenAI, Qdrant, Neon)
2. ✅ Deploy backend to Railway
3. ✅ Generate embeddings
4. ✅ Deploy frontend to GitHub Pages
5. ✅ Test everything
6. ✅ Create demo video
7. ✅ Submit to hackathon

**Good luck! 🚀**
