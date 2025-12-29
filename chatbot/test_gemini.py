"""
Test Gemini API connection and functionality
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

def test_gemini():
    """Test Gemini service"""

    print("=" * 70)
    print("TESTING GEMINI API (FREE)")
    print("=" * 70)

    print("\n[1/4] Testing Gemini Service Import...")
    try:
        from api.services.gemini_service import get_gemini_service
        print("  [OK] Gemini service imported successfully")
    except Exception as e:
        print(f"  [ERROR] Failed to import: {e}")
        return False

    print("\n[2/4] Initializing Gemini Service...")
    try:
        gemini_svc = get_gemini_service()
        print("  [OK] Gemini service initialized")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize: {e}")
        print(f"  [HINT] Check your GEMINI_API_KEY in .env file")
        return False

    print("\n[3/4] Testing Embedding Generation...")
    try:
        test_text = "ROS 2 is a robot operating system"
        embedding = gemini_svc.generate_embedding(test_text)
        print(f"  [OK] Generated embedding: {len(embedding)} dimensions")
        print(f"  [INFO] Gemini uses 768-dimensional vectors")
    except Exception as e:
        print(f"  [ERROR] Failed to generate embedding: {e}")
        return False

    print("\n[4/4] Testing Chat Completion...")
    try:
        response = gemini_svc.chat_completion(
            messages=[
                {"role": "user", "content": "What is ROS 2? Answer in one sentence."}
            ],
            temperature=0.7,
            max_tokens=100
        )
        print(f"  [OK] Generated response:")
        print(f"  {response[:200]}...")
    except Exception as e:
        print(f"  [ERROR] Failed to generate response: {e}")
        return False

    print("\n" + "=" * 70)
    print("ALL GEMINI TESTS PASSED!")
    print("=" * 70)

    print("\n[SUCCESS] Gemini is working perfectly!")
    print("\n100% FREE AI Service:")
    print("  - Embedding: 768 dimensions")
    print("  - Chat: Gemini Pro model")
    print("  - Cost: $0.00")
    print("  - Rate limit: 1M tokens/min")

    print("\n[NEXT STEPS]")
    print("1. Generate embeddings:")
    print("   npm run generate:embeddings")
    print("\n2. Start chatbot API:")
    print("   npm run chatbot:start")
    print("\n3. Start frontend:")
    print("   npm start")

    return True

if __name__ == "__main__":
    success = test_gemini()
    sys.exit(0 if success else 1)
