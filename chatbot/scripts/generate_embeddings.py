"""
Script to generate embeddings for all textbook chapters and upload to Qdrant
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.services.gemini_service import get_gemini_service
from api.services.qdrant_service import get_qdrant_service
from dotenv import load_dotenv
import os

load_dotenv()

# Check which provider to use for embeddings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

if EMBEDDING_PROVIDER == "gemini":
    get_embedding_service = get_gemini_service
    print("Using Gemini for embeddings")
else:
    from api.services.openai_service import get_openai_service
    get_embedding_service = get_openai_service
    print("Using OpenAI for embeddings")

class ChunkProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: dict) -> List[Dict]:
        """
        Split text into overlapping chunks

        Args:
            text: Text to chunk
            metadata: Metadata for the text (chapter, section, etc.)

        Returns:
            List of chunks with metadata
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para.split())

            if current_size + para_size > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        **metadata
                    })

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-self.chunk_overlap:])
                    current_chunk = overlap_text + "\n\n" + para
                    current_size = len(overlap_text.split()) + para_size
                else:
                    current_chunk = para
                    current_size = para_size
            else:
                current_chunk += "\n\n" + para
                current_size += para_size

        # Add final chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                **metadata
            })

        return chunks

def parse_markdown_file(file_path: Path) -> Dict:
    """
    Parse markdown file to extract metadata and content

    Args:
        file_path: Path to markdown file

    Returns:
        Dict with chapter_id, chapter_title, sections
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract chapter title (first # heading)
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    chapter_title = title_match.group(1) if title_match else file_path.stem

    # Generate chapter ID from file path
    # e.g., docs/module-1/ros2-fundamentals.md -> module-1/ros2-fundamentals
    relative_path = file_path.relative_to(file_path.parents[2])
    chapter_id = str(relative_path.with_suffix('')).replace('\\', '/')

    # Split by sections (## headings)
    sections = re.split(r'\n##\s+', content)

    parsed_sections = []
    for i, section in enumerate(sections):
        if i == 0:
            # First section (before first ##)
            section_title = "Introduction"
            section_content = section
        else:
            # Extract section title and content
            lines = section.split('\n', 1)
            section_title = lines[0].strip()
            section_content = lines[1] if len(lines) > 1 else ""

        # Remove code blocks for embedding (keep explanatory text)
        # Keep code comments for context
        text_only = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK]', section_content)

        # Remove excessive whitespace
        text_only = re.sub(r'\n{3,}', '\n\n', text_only)

        parsed_sections.append({
            "section_title": section_title,
            "content": text_only.strip()
        })

    return {
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "sections": parsed_sections
    }

def generate_embeddings_for_docs(docs_dir: str = "../docs"):
    """Generate and upload embeddings for all markdown files"""

    ai_svc = get_ai_service()
    qdrant_svc = get_qdrant_service()
    chunker = ChunkProcessor(chunk_size=500, chunk_overlap=50)

    print(f"Using AI Provider: {AI_PROVIDER.upper()}")

    # Initialize Qdrant collection
    qdrant_svc.create_collection()

    docs_path = Path(__file__).parent.parent.parent / "docs"

    # Find all markdown files
    markdown_files = list(docs_path.rglob("*.md"))

    # Filter out tutorial files
    markdown_files = [
        f for f in markdown_files
        if "tutorial" not in str(f).lower()
    ]

    print(f"Found {len(markdown_files)} markdown files")

    all_chunks = []
    all_vectors = []
    all_payloads = []

    for file_path in markdown_files:
        print(f"\nProcessing: {file_path.name}")

        try:
            parsed = parse_markdown_file(file_path)

            for section in parsed["sections"]:
                if not section["content"].strip():
                    continue

                # Create chunks for this section
                section_chunks = chunker.chunk_text(
                    section["content"],
                    {
                        "chapter_id": parsed["chapter_id"],
                        "chapter_title": parsed["chapter_title"],
                        "section_title": section["section_title"]
                    }
                )

                all_chunks.extend(section_chunks)

            print(f"  - Created {len(section_chunks)} chunks")

        except Exception as e:
            print(f"  - Error: {e}")
            continue

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Generate embeddings in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [chunk["content"] for chunk in batch]

        print(f"Generating embeddings for batch {i//batch_size + 1}...")

        try:
            vectors = ai_svc.generate_embeddings_batch(texts)
            all_vectors.extend(vectors)
            all_payloads.extend(batch)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Try one by one
            for j, text in enumerate(texts):
                try:
                    vector = ai_svc.generate_embedding(text)
                    all_vectors.append(vector)
                    all_payloads.append(batch[j])
                except Exception as e2:
                    print(f"Skipping chunk: {e2}")

    # Upload to Qdrant
    print(f"\nUploading {len(all_vectors)} vectors to Qdrant...")

    try:
        qdrant_svc.upsert_vectors(
            vectors=all_vectors,
            payloads=all_payloads
        )
        print("✅ Successfully uploaded all vectors!")

        # Print collection info
        info = qdrant_svc.get_collection_info()
        print(f"\nCollection Info:")
        print(f"  - Vectors: {info.get('vectors_count', 0)}")
        print(f"  - Points: {info.get('points_count', 0)}")

    except Exception as e:
        print(f"❌ Error uploading vectors: {e}")

if __name__ == "__main__":
    print("🚀 Starting embedding generation...\n")
    generate_embeddings_for_docs()
    print("\n✅ Done!")
