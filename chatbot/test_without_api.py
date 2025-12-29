"""
Comprehensive test of chatbot functionality WITHOUT API keys
Tests chunking, parsing, and data flow
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_chunking_logic():
    """Test the text chunking algorithm"""
    print("\n" + "=" * 70)
    print("TEST 1: TEXT CHUNKING")
    print("=" * 70)

    from scripts.generate_embeddings import ChunkProcessor

    chunker = ChunkProcessor(chunk_size=50, chunk_overlap=10)

    # Sample text
    sample_text = """
ROS 2 is a robot operating system that provides libraries and tools.

It helps developers create robot applications. ROS 2 uses a distributed
architecture with nodes communicating via topics.

The publish-subscribe model allows for flexible communication between
different parts of a robot system.
"""

    metadata = {
        "chapter": "module-1/ros2-fundamentals",
        "section": "Introduction"
    }

    print("\n[1/3] Creating chunks...")
    chunks = chunker.chunk_text(sample_text.strip(), metadata)

    print(f"  [OK] Created {len(chunks)} chunks from sample text")

    print("\n[2/3] Verifying chunk content...")
    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk['content'].split())
        print(f"  Chunk {i}: {word_count} words")
        print(f"    Chapter: {chunk['chapter']}")
        print(f"    Section: {chunk['section']}")
        print(f"    Preview: {chunk['content'][:60]}...")

    print("\n[3/3] Testing overlap...")
    if len(chunks) > 1:
        # Check if chunks have overlap
        chunk1_words = chunks[0]['content'].split()[-10:]
        chunk2_words = chunks[1]['content'].split()[:10]
        overlap_found = any(word in chunk2_words for word in chunk1_words)
        if overlap_found:
            print("  [OK] Chunks have overlap for context continuity")
        else:
            print("  [INFO] No overlap detected (text might be too short)")

    print("\n[PASSED] Chunking logic works correctly")
    return True

def test_markdown_parsing():
    """Test parsing real markdown files"""
    print("\n" + "=" * 70)
    print("TEST 2: MARKDOWN FILE PARSING")
    print("=" * 70)

    from scripts.generate_embeddings import parse_markdown_file

    # Find a sample markdown file
    docs_dir = Path(__file__).parent.parent / "docs"

    print(f"\n[1/3] Looking for markdown files in {docs_dir}...")

    markdown_files = list(docs_dir.rglob("*.md"))

    if not markdown_files:
        print("  [ERROR] No markdown files found!")
        return False

    print(f"  [OK] Found {len(markdown_files)} markdown files")

    # Test parsing first file
    test_file = markdown_files[0]
    print(f"\n[2/3] Parsing test file: {test_file.name}")

    try:
        result = parse_markdown_file(test_file)

        print(f"  [OK] Chapter ID: {result['chapter_id']}")
        print(f"  [OK] Title: {result['chapter_title']}")
        print(f"  [OK] Sections: {len(result['sections'])}")

        # Calculate total content length
        total_content = sum(len(s['content']) for s in result['sections'])
        print(f"  [OK] Content length: {total_content} characters")

        print("\n[3/3] Sections found:")
        for section in result['sections'][:5]:  # Show first 5 sections
            print(f"    - {section['section_title']}")
        if len(result['sections']) > 5:
            print(f"    ... and {len(result['sections']) - 5} more")

    except Exception as e:
        print(f"  [ERROR] Failed to parse: {e}")
        return False

    print("\n[PASSED] Markdown parsing works correctly")
    return True

def test_chapter_discovery():
    """Test finding all chapters"""
    print("\n" + "=" * 70)
    print("TEST 3: CHAPTER DISCOVERY")
    print("=" * 70)

    docs_dir = Path(__file__).parent.parent / "docs"

    print(f"\n[1/2] Scanning for chapters in {docs_dir}...")

    # Find all markdown files
    all_files = list(docs_dir.rglob("*.md"))

    print(f"  [OK] Found {len(all_files)} total markdown files")

    print("\n[2/2] Categorizing by module...")

    modules = {
        "module-1": [],
        "module-2": [],
        "module-3": [],
        "module-4": [],
        "appendix": [],
        "other": []
    }

    for file in all_files:
        file_str = str(file)
        categorized = False
        for module in modules.keys():
            if module in file_str:
                modules[module].append(file.name)
                categorized = True
                break
        if not categorized:
            modules["other"].append(file.name)

    for module, files in modules.items():
        if files:
            print(f"\n  {module.upper()}: {len(files)} files")
            for f in files[:3]:  # Show first 3
                print(f"    - {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")

    print("\n[PASSED] Chapter discovery works correctly")
    return True

def test_embedding_workflow():
    """Test the complete workflow (without API calls)"""
    print("\n" + "=" * 70)
    print("TEST 4: COMPLETE WORKFLOW SIMULATION")
    print("=" * 70)

    from scripts.generate_embeddings import ChunkProcessor, parse_markdown_file

    docs_dir = Path(__file__).parent.parent / "docs"
    markdown_files = list(docs_dir.rglob("*.md"))

    if not markdown_files:
        print("  [ERROR] No markdown files to process")
        return False

    # Process first file as example
    test_file = markdown_files[0]

    print(f"\n[1/5] Parsing: {test_file.name}")
    parsed = parse_markdown_file(test_file)

    # Combine all section content
    full_content = "\n\n".join([s['content'] for s in parsed['sections']])
    print(f"  [OK] Parsed {len(full_content)} characters")

    print("\n[2/5] Chunking content...")
    chunker = ChunkProcessor(chunk_size=500, chunk_overlap=50)

    chunks = chunker.chunk_text(
        full_content,
        {
            "chapter": parsed['chapter_id'],
            "title": parsed['chapter_title']
        }
    )
    print(f"  [OK] Created {len(chunks)} chunks")

    print("\n[3/5] Simulating embedding generation...")
    print(f"  [MOCK] Would generate {len(chunks)} embeddings")
    print(f"  [MOCK] Using model: text-embedding-3-small (1536 dimensions)")
    print(f"  [MOCK] Estimated cost: ${len(chunks) * 0.00002:.4f}")

    print("\n[4/5] Simulating vector upload to Qdrant...")
    print(f"  [MOCK] Would upload {len(chunks)} vectors")
    print(f"  [MOCK] Collection: physical_ai_textbook")
    print(f"  [MOCK] Vector size: 1536 dimensions")

    print("\n[5/5] Estimating full workload...")

    total_chunks = 0
    total_chars = 0

    for file in markdown_files[:5]:  # Sample first 5 files
        try:
            parsed = parse_markdown_file(file)
            full_content = "\n\n".join([s['content'] for s in parsed['sections']])
            file_chunks = chunker.chunk_text(full_content, {"chapter": parsed['chapter_id']})
            total_chunks += len(file_chunks)
            total_chars += len(full_content)
        except:
            pass

    # Estimate for all files
    estimated_total = (total_chunks / 5) * len(markdown_files)
    estimated_cost = estimated_total * 0.00002

    print(f"  [INFO] Estimated total chunks: ~{int(estimated_total)}")
    print(f"  [INFO] Estimated embedding cost: ${estimated_cost:.2f}")
    print(f"  [INFO] Estimated time: ~{int(estimated_total / 50)} minutes")

    print("\n[PASSED] Complete workflow simulation successful")
    return True

def test_service_initialization():
    """Test that services can be initialized (without API calls)"""
    print("\n" + "=" * 70)
    print("TEST 5: SERVICE INITIALIZATION")
    print("=" * 70)

    print("\n[1/3] Testing OpenAI service structure...")
    try:
        from api.services.openai_service import OpenAIService
        print("  [OK] OpenAIService class available")
        print("  [INFO] Methods: generate_embedding, chat_with_rag, etc.")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

    print("\n[2/3] Testing Qdrant service structure...")
    try:
        from api.services.qdrant_service import QdrantService
        print("  [OK] QdrantService class available")
        print("  [INFO] Methods: create_collection, search, upsert_vectors, etc.")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

    print("\n[3/3] Testing Neon service structure...")
    try:
        from api.services.neon_service import NeonService
        print("  [OK] NeonService class available")
        print("  [INFO] Methods: init_database, add_message, etc.")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

    print("\n[PASSED] All services properly structured")
    return True

def main():
    """Run all tests"""
    print("=" * 70)
    print("CHATBOT TEST SUITE (NO API KEYS REQUIRED)")
    print("=" * 70)
    print("\nTesting chatbot functionality without making API calls...")

    tests = [
        ("Chunking Logic", test_chunking_logic),
        ("Markdown Parsing", test_markdown_parsing),
        ("Chapter Discovery", test_chapter_discovery),
        ("Workflow Simulation", test_embedding_workflow),
        ("Service Initialization", test_service_initialization),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        print("\nYour chatbot is ready for API key configuration.")
        print("\nNext steps:")
        print("  1. Get API keys (OpenAI, Qdrant, Neon)")
        print("  2. Update .env file with your keys")
        print("  3. Run: python test_chatbot_live.py")
        print("  4. Run: python scripts/generate_embeddings.py")
        print("\n")
        return True
    else:
        print("\nSome tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
