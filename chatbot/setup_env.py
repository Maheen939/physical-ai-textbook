"""
Interactive script to help set up .env file for chatbot
"""

import os
from pathlib import Path

def setup_env():
    """Guide user through setting up .env file"""

    print("=" * 70)
    print("CHATBOT ENVIRONMENT SETUP")
    print("=" * 70)
    print("\nThis script will help you create a .env file with your API keys.")
    print("\nYou'll need:")
    print("  1. OpenAI API Key - https://platform.openai.com/api-keys")
    print("  2. Qdrant Cloud - https://cloud.qdrant.io/")
    print("  3. Neon Postgres - https://neon.tech/")

    print("\n" + "-" * 70)
    print("OPTION 1: Manual Setup (Recommended)")
    print("-" * 70)
    print("\n1. Copy .env.example to .env:")
    print("   copy .env.example .env")
    print("\n2. Edit .env and add your API keys")
    print("\n3. Run: python scripts/generate_embeddings.py")

    print("\n" + "-" * 70)
    print("OPTION 2: Quick Test (Mock Mode)")
    print("-" * 70)
    print("\nIf you don't have API keys yet, you can test the structure:")
    print("   python simple_test.py")

    print("\n" + "-" * 70)
    print("GETTING API KEYS")
    print("-" * 70)

    print("\n[1] OpenAI API Key:")
    print("    - Go to: https://platform.openai.com/api-keys")
    print("    - Sign up / Login")
    print("    - Click 'Create new secret key'")
    print("    - Copy key (starts with 'sk-')")
    print("    - Add $5-10 credit to your account")
    print("    Cost: ~$0.05 per query, $1-2 for initial embeddings")

    print("\n[2] Qdrant Cloud (Vector Database):")
    print("    - Go to: https://cloud.qdrant.io/")
    print("    - Sign up (NO credit card required)")
    print("    - Click 'Create Cluster'")
    print("    - Name: physical-ai-textbook")
    print("    - Plan: FREE (1GB storage)")
    print("    - Copy: Cluster URL and API Key")
    print("    Cost: FREE")

    print("\n[3] Neon Postgres (Conversation Storage):")
    print("    - Go to: https://neon.tech/")
    print("    - Sign up (NO credit card required)")
    print("    - Click 'Create Project'")
    print("    - Name: physical-ai-chatbot")
    print("    - Plan: FREE (0.5GB storage)")
    print("    - Copy: Connection String")
    print("    Cost: FREE")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Get the 3 API keys above")
    print("2. Create .env file: copy .env.example .env")
    print("3. Edit .env with your keys")
    print("4. Generate embeddings: python scripts/generate_embeddings.py")
    print("5. Start API server: python api/main.py")
    print("\nFor detailed instructions, see: DEPLOYMENT.md")
    print("\n")

if __name__ == "__main__":
    setup_env()
