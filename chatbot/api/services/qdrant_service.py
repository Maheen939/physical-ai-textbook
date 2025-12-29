"""
Qdrant Vector Database Service for semantic search
"""

import os
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

class QdrantService:
    def __init__(self):
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = "physical_ai_textbook"

        # Determine vector size based on AI provider
        ai_provider = os.getenv("AI_PROVIDER", "gemini").lower()
        self.vector_size = 768 if ai_provider == "gemini" else 1536  # Gemini: 768, OpenAI: 1536

    def create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")

        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def upsert_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict],
        ids: Optional[List[int]] = None
    ):
        """
        Insert or update vectors in Qdrant

        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dicts
            ids: Optional list of IDs (will auto-generate if not provided)
        """
        try:
            if ids is None:
                # Auto-generate IDs
                ids = list(range(len(vectors)))

            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            print(f"Upserted {len(points)} vectors to {self.collection_name}")

        except Exception as e:
            print(f"Error upserting vectors: {e}")
            raise

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        chapter_filter: Optional[str] = None,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            chapter_filter: Filter by chapter ID (optional)
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of search results with content and metadata
        """
        try:
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "score_threshold": score_threshold
            }

            # Add chapter filter if specified
            if chapter_filter:
                search_params["query_filter"] = Filter(
                    must=[
                        FieldCondition(
                            key="chapter_id",
                            match=MatchValue(value=chapter_filter)
                        )
                    ]
                )

            results = self.client.search(**search_params)

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.payload.get("content", ""),
                    "chapter": result.payload.get("chapter_title", ""),
                    "chapter_id": result.payload.get("chapter_id", ""),
                    "section": result.payload.get("section_title", ""),
                    "score": result.score,
                    "metadata": result.payload
                })

            return formatted_results

        except Exception as e:
            print(f"Error searching vectors: {e}")
            raise

    def delete_collection(self):
        """Delete the collection (use with caution!)"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
            raise

    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}

# Singleton instance
_qdrant_service = None

def get_qdrant_service() -> QdrantService:
    """Get or create Qdrant service singleton"""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
