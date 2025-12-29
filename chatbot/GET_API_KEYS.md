# How to Get FREE API Keys for Chatbot

This guide will walk you through getting all required API keys. **2 out of 3 are completely FREE!**

---

## 🆓 Step 1: Qdrant Cloud (FREE - Vector Database)

**Cost**: FREE Forever (1GB storage, 1M vectors)
**Time**: 3-5 minutes
**Credit Card**: NOT Required ✅

### Instructions:

1. **Go to Qdrant Cloud**
   - Visit: https://cloud.qdrant.io/
   - Click "Sign Up" or "Get Started"

2. **Create Account**
   - Sign up with Google/GitHub, or use email
   - Verify your email if needed
   - No credit card required!

3. **Create a Cluster**
   - Click "Create Cluster" button
   - Fill in:
     - **Name**: `physical-ai-textbook` (or any name you prefer)
     - **Region**: Choose closest to you (e.g., `us-east`, `eu-west`)
     - **Plan**: Select **"Free"** (1GB, 1M vectors)
   - Click "Create"
   - Wait 1-2 minutes for cluster to initialize

4. **Get Your Credentials**

   **Cluster URL**:
   - You'll see it on the cluster page
   - Format: `https://xxxxx-xxxxx.qdrant.io`
   - Example: `https://a1b2c3d4-e5f6.us-east-1-0.aws.cloud.qdrant.io`
   - Copy this entire URL

   **API Key**:
   - Click on your cluster name
   - Go to "API Keys" tab or "Data Access" section
   - Click "Create API Key" or copy the existing key
   - Format: Long alphanumeric string
   - Copy and save this securely

5. **Save These Values**:
   ```
   QDRANT_URL=https://xxxxx-xxxxx.qdrant.io
   QDRANT_API_KEY=your-long-api-key-here
   ```

---

## 🆓 Step 2: Neon Postgres (FREE - Database)

**Cost**: FREE Forever (0.5GB storage)
**Time**: 3-5 minutes
**Credit Card**: NOT Required ✅

### Instructions:

1. **Go to Neon**
   - Visit: https://neon.tech/
   - Click "Sign Up" or "Get Started"

2. **Create Account**
   - Sign up with Google/GitHub, or use email
   - Verify your email if needed
   - No credit card required!

3. **Create a Project**
   - You'll be prompted to create your first project
   - Fill in:
     - **Project Name**: `physical-ai-chatbot` (or any name)
     - **Region**: Choose closest to you
     - **Plan**: **Free** (automatically selected)
   - Click "Create Project"

4. **Get Your Database URL**
   - After project is created, you'll see a connection string
   - Click "Copy" next to the connection string
   - Format: `postgresql://username:password@hostname/database?sslmode=require`
   - Example: `postgresql://user_abc123:pass_xyz789@ep-cool-name-123456.us-east-2.aws.neon.tech/neondb?sslmode=require`

5. **Important Notes**:
   - The password is only shown ONCE during creation
   - If you lose it, you can reset it from Settings → Reset password
   - Make sure to copy the FULL connection string including `?sslmode=require`

6. **Save This Value**:
   ```
   DATABASE_URL=postgresql://user:pass@host/database?sslmode=require
   ```

---

## 💳 Step 3: OpenAI API Key (PAID - ~$1-2 for embeddings)

**Cost**: ~$1-2 one-time for embeddings, then ~$0.05 per query
**Time**: 5-10 minutes
**Credit Card**: Required (you'll add $5-10 credit)

### Instructions:

1. **Go to OpenAI Platform**
   - Visit: https://platform.openai.com/
   - Click "Sign Up" (or "Log In" if you have ChatGPT account)

2. **Create/Login to Account**
   - Sign up with Google, Microsoft, or email
   - Complete verification if needed

3. **Add Billing Information**
   - Go to: https://platform.openai.com/account/billing
   - Click "Add payment method"
   - Add your credit/debit card
   - **Add Credits**: Add $5-10 to start
     - Go to "Billing" → "Add to credit balance"
     - Enter $5 or $10 (should last for months)

4. **Create API Key**
   - Go to: https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Give it a name: `physical-ai-textbook`
   - Click "Create secret key"
   - **⚠️ IMPORTANT**: Copy the key NOW - it's only shown once!
   - Format: `sk-proj-...` (starts with `sk-`)

5. **Cost Breakdown**:
   - **Embedding Generation** (one-time): ~$1-2
     - ~450 chunks × 500 words each
     - Model: `text-embedding-3-small`
     - Cost: $0.00002 per 1K tokens

   - **Chat Queries** (ongoing): ~$0.05 per query
     - Uses GPT-4 for responses
     - ~5 context chunks per query
     - Most students ask 10-20 questions = $0.50-$1.00 total

6. **Save This Value**:
   ```
   OPENAI_API_KEY=sk-proj-your-key-here
   ```

---

## 📝 Configuration Summary

Once you have all three keys, you'll have:

```bash
# From Qdrant Cloud (FREE)
QDRANT_URL=https://xxxxx-xxxxx.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# From Neon (FREE)
DATABASE_URL=postgresql://user:pass@host/database?sslmode=require

# From OpenAI (PAID ~$1-2)
OPENAI_API_KEY=sk-proj-your-openai-key
```

---

## 🔐 Security Notes

1. **Never commit API keys to Git**
   - The `.env` file is already in `.gitignore`
   - Never share your keys publicly

2. **Keep keys secure**
   - Store them in a password manager
   - Don't share in screenshots or videos

3. **Monitor usage**
   - Check OpenAI usage: https://platform.openai.com/usage
   - Qdrant shows storage usage in dashboard
   - Neon shows database size in dashboard

---

## ✅ Next Steps

After getting all three keys:

1. **Open**: `D:\textbook\physical-ai-textbook\chatbot\.env`

2. **Fill in your keys**:
   ```bash
   OPENAI_API_KEY=sk-proj-your-key-here
   QDRANT_URL=https://xxxxx.qdrant.io
   QDRANT_API_KEY=your-qdrant-key
   DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
   ```

3. **Save the file**

4. **Test connections**:
   ```bash
   npm run chatbot:test-live
   ```

5. **Generate embeddings**:
   ```bash
   npm run generate:embeddings
   ```

---

## 🆘 Troubleshooting

### Qdrant Issues
- **"Cluster not ready"**: Wait 2-3 minutes after creation
- **"Authentication failed"**: Check API key has no extra spaces
- **"Connection refused"**: Make sure URL includes `https://`

### Neon Issues
- **"Connection refused"**: Make sure `?sslmode=require` is at end of URL
- **"Authentication failed"**: Password might be wrong - reset it
- **"Database not found"**: Check database name in URL matches your project

### OpenAI Issues
- **"Insufficient credits"**: Add more funds to your account
- **"Invalid API key"**: Make sure key starts with `sk-`
- **"Rate limit exceeded"**: Wait a minute and try again

---

## 💰 Total Cost Breakdown

| Service | Setup | Monthly | Notes |
|---------|-------|---------|-------|
| **Qdrant** | $0 | $0 | FREE forever (1GB) |
| **Neon** | $0 | $0 | FREE forever (0.5GB) |
| **OpenAI** | $1-2 | $0-5 | One-time embeddings + queries |
| **TOTAL** | **$1-2** | **$0-5** | Very affordable! |

---

## 📞 Support

If you have issues getting API keys:

- **Qdrant**: https://qdrant.tech/support/
- **Neon**: https://neon.tech/docs/
- **OpenAI**: https://help.openai.com/

Good luck! 🚀
