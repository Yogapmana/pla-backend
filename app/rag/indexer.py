import logging
from io import BytesIO
from app.rag.chunker import chunk_markdown
from app.rag.embedder import embed_texts
from app.rag.vector_store import get_qdrant_client, upsert_chunks

logger = logging.getLogger(__name__)


def index_module(
    user_id: str,
    session_id: str,
    topic_id: str,
    module_id: str,
    title: str,
    content_markdown: str,
    week: int = 1,
    day: int = 1,
    sources: list[dict] | None = None,
) -> int:
    """
    Full indexing pipeline:
    1. Chunk the markdown content
    2. Embed chunks with nomic-embed-text
    3. Upsert into Qdrant under user's collection
    Returns the number of chunks indexed.
    """
    logger.info(f"[INDEXER] Chunking module '{title}' for topic {topic_id}...")
    chunks = chunk_markdown(content_markdown)
    if not chunks:
        logger.warning(f"[INDEXER] No chunks produced for module '{title}'. Skipping.")
        return 0

    logger.info(f"[INDEXER] Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)

    metadata = {
        "user_id": user_id,
        "session_id": session_id,
        "topic_id": topic_id,
        "module_id": module_id,
        "week": week,
        "day": day,
        "source_type": "module",
        "source_title": title,
        "source_url": "",
    }

    client = get_qdrant_client()
    count = upsert_chunks(
        client=client,
        user_id=user_id,
        chunks=chunks,
        embeddings=embeddings,
        metadata=metadata,
    )
    logger.info(f"[INDEXER] Indexed {count} chunks into Qdrant for user {user_id}.")
    return count


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF, TXT, or MD files."""
    text = ""
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        # Same robust extractor as onboarding (pdftotext, fallback PyMuPDF).
        from app.utils.pdf_extractor import extract_pdf_text
        text = extract_pdf_text(file_bytes)
    elif filename_lower.endswith((".txt", ".md")):
        # Decode as utf-8
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    return text.strip()


def index_uploaded_document(
    user_id: str,
    session_id: str,
    topic_id: str,
    filename: str,
    raw_text: str,
) -> int:
    """
    Index an uploaded user document into Qdrant.
    """
    logger.info(f"[INDEXER] Chunking uploaded document '{filename}'...")
    chunks = chunk_markdown(raw_text)
    if not chunks:
        logger.warning(f"[INDEXER] No chunks produced for document '{filename}'. Skipping.")
        return 0

    logger.info(f"[INDEXER] Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)

    metadata = {
        "user_id": user_id,
        "session_id": session_id,
        "topic_id": topic_id,
        "source_type": "user_upload",
        "source_title": filename,
        "source_url": "",
    }

    client = get_qdrant_client()
    count = upsert_chunks(
        client=client,
        user_id=user_id,
        chunks=chunks,
        embeddings=embeddings,
        metadata=metadata,
    )
    logger.info(f"[INDEXER] Indexed {count} chunks of document '{filename}' into Qdrant for user {user_id}.")
    return count
