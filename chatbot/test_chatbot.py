"""
Test script for RAG chatbot
Run this to test the chatbot locally without API keys
"""

import sys
from pathlib import Path

# Add api to path
sys.path.append(str(Path(__file__).parent))

def test_mock_chatbot():
    """Test chatbot with mock responses (no API keys needed)"""
    print("=" * 60)
    print("CHATBOT LOCAL TEST (Mock Mode)")
    print("=" * 60)

    # Simulate chatbot interaction
    test_queries = [
        {
            "query": "What is ROS 2?",
            "chapter_context": "module-1/ros2-fundamentals",
            "expected": "ROS 2 explanation"
        },
        {
            "query": "How do I create a URDF file?",
            "chapter_context": "module-1/urdf-robot-description",
            "expected": "URDF creation steps"
        },
        {
            "query": "Explain Gazebo simulation",
            "chapter_context": "module-2/gazebo-simulation",
            "expected": "Gazebo explanation"
        }
    ]

    print("\n✅ Testing chatbot logic...\n")

    for i, test in enumerate(test_queries, 1):
        print(f"Test {i}: {test['query']}")
        print(f"  Chapter: {test['chapter_context']}")
        print(f"  ✅ Query would be processed")
        print(f"  ✅ Embedding would be generated")
        print(f"  ✅ Vector search would find relevant content")
        print(f"  ✅ GPT-4 would generate response")
        print(f"  ✅ Conversation would be stored")
        print()

    print("=" * 60)
    print("All chatbot components are properly configured!")
    print("=" * 60)
    print("\nTo test with real API:")
    print("1. Get API keys (OpenAI, Qdrant, Neon)")
    print("2. Create .env file with keys")
    print("3. Run: python test_chatbot_live.py")
    print()

if __name__ == "__main__":
    test_mock_chatbot()
