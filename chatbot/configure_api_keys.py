"""
Interactive script to configure API keys for the chatbot
"""

import os
from pathlib import Path

def configure_api_keys():
    """Interactively collect and save API keys"""

    print("=" * 70)
    print("CHATBOT API KEY CONFIGURATION")
    print("=" * 70)
    print("\nThis script will help you configure your API keys.")
    print("Your keys will be saved to .env file (which is git-ignored).")

    print("\n" + "-" * 70)
    print("REQUIRED API KEYS")
    print("-" * 70)

    # Collect keys
    keys = {}

    print("\n[1/4] OpenAI API Key")
    print("  Get from: https://platform.openai.com/api-keys")
    print("  Format: sk-...")
    openai_key = input("\n  Enter your OpenAI API key: ").strip()
    keys['OPENAI_API_KEY'] = openai_key

    print("\n[2/4] Qdrant Cloud URL")
    print("  Get from: https://cloud.qdrant.io/")
    print("  Format: https://xxxxx.qdrant.io")
    qdrant_url = input("\n  Enter your Qdrant URL: ").strip()
    keys['QDRANT_URL'] = qdrant_url

    print("\n[3/4] Qdrant API Key")
    print("  Get from: https://cloud.qdrant.io/ (same page as URL)")
    qdrant_key = input("\n  Enter your Qdrant API key: ").strip()
    keys['QDRANT_API_KEY'] = qdrant_key

    print("\n[4/4] Neon Postgres Database URL")
    print("  Get from: https://neon.tech/")
    print("  Format: postgresql://user:pass@host/database")
    db_url = input("\n  Enter your Neon Database URL: ").strip()
    keys['DATABASE_URL'] = db_url

    # Validate inputs
    print("\n" + "-" * 70)
    print("VALIDATING INPUTS")
    print("-" * 70)

    valid = True

    if not openai_key.startswith('sk-'):
        print("  [WARNING] OpenAI key should start with 'sk-'")
        valid = False
    else:
        print("  [OK] OpenAI key format looks good")

    if not qdrant_url.startswith('https://') or not 'qdrant.io' in qdrant_url:
        print("  [WARNING] Qdrant URL should be https://xxxxx.qdrant.io")
        valid = False
    else:
        print("  [OK] Qdrant URL format looks good")

    if len(qdrant_key) < 10:
        print("  [WARNING] Qdrant API key seems too short")
        valid = False
    else:
        print("  [OK] Qdrant API key format looks good")

    if not db_url.startswith('postgresql://'):
        print("  [WARNING] Database URL should start with postgresql://")
        valid = False
    else:
        print("  [OK] Database URL format looks good")

    if not valid:
        print("\n[WARNING] Some keys might be invalid. Continue anyway? (y/n)")
        confirm = input("  > ").strip().lower()
        if confirm != 'y':
            print("\nConfiguration cancelled. Please check your keys and try again.")
            return False

    # Write to .env file
    print("\n" + "-" * 70)
    print("SAVING TO .ENV FILE")
    print("-" * 70)

    env_content = f"""# OpenAI API Key
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY={keys['OPENAI_API_KEY']}

# Qdrant Cloud Configuration
# Get from: https://cloud.qdrant.io/
QDRANT_URL={keys['QDRANT_URL']}
QDRANT_API_KEY={keys['QDRANT_API_KEY']}

# Neon Postgres Database URL
# Get from: https://neon.tech/
DATABASE_URL={keys['DATABASE_URL']}

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4
TOP_K_RESULTS=5

# Optional: Frontend URL for CORS
FRONTEND_URL=http://localhost:3000
"""

    env_path = Path(__file__).parent / ".env"

    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print(f"\n  [OK] Configuration saved to: {env_path}")
    except Exception as e:
        print(f"\n  [ERROR] Failed to save: {e}")
        return False

    # Success
    print("\n" + "=" * 70)
    print("CONFIGURATION COMPLETE!")
    print("=" * 70)

    print("\nYour API keys have been configured successfully!")

    print("\n" + "-" * 70)
    print("NEXT STEPS")
    print("-" * 70)
    print("\n1. Test your API connections:")
    print("   python test_chatbot_live.py")
    print("\n2. Generate embeddings (10-15 minutes, ~$1-2):")
    print("   python scripts/generate_embeddings.py")
    print("\n3. Start the API server:")
    print("   python api/main.py")
    print("\n4. Start the frontend:")
    print("   cd .. && npm start")
    print("\n")

    return True

if __name__ == "__main__":
    try:
        success = configure_api_keys()
        if success:
            print("Ready to generate embeddings!")
    except KeyboardInterrupt:
        print("\n\nConfiguration cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
