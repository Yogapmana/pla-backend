import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.config import settings

VECTOR_SIZE = 768  # nomic-embed-text dimension

def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client instance."""
    return QdrantClient(url=settings.QDRANT_URL)

def get_collection_name(user_id: str) -> str:
    """Get collection name scoped per user."""
    return f"pla_user_{user_id.replace('-', '_')}"

def ensure_collection(client: QdrantClient, user_id: str) -> str:
    """Create the Qdrant collection for a user if it doesn't exist."""
    collection_name = get_collection_name(user_id)
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    return collection_name

def upsert_chunks(
    client: QdrantClient,
    user_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: dict,
) -> int:
    """Upsert chunk embeddings with payload metadata into Qdrant."""
    collection_name = ensure_collection(client, user_id)
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        payload = {
            **metadata,
            "chunk_index": i,
            "text": chunk,
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=payload,
        ))
    client.upsert(collection_name=collection_name, points=points)
    return len(points)

def search_chunks(
    client: QdrantClient,
    user_id: str,
    query_vector: list[float],
    session_id: str | None = None,
    topic_id: str | None = None,
    top_k: int = 8,
) -> list[dict]:
    """Search for similar chunks in Qdrant, optionally filtered by session_id and topic_id."""
    collection_name = get_collection_name(user_id)
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        return []

    must_conditions = []
    if session_id:
        must_conditions.append(FieldCondition(key="session_id", match=MatchValue(value=str(session_id))))
    if topic_id:
        must_conditions.append(FieldCondition(key="topic_id", match=MatchValue(value=str(topic_id))))

    query_filter = None
    if must_conditions:
        query_filter = Filter(must=must_conditions)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": r.score, **r.payload} for r in results.points]
