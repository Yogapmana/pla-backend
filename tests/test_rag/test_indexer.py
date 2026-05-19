"""
Test RAG Indexing Pipeline

Tests chunking, embedding (nomic-embed-text via Ollama), and Qdrant upsert.
Run with: PYTHONPATH=. ./venv/bin/python tests/test_rag/test_indexer.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

SAMPLE_MARKDOWN = """
# Introduction to Python

Python adalah bahasa pemrograman tingkat tinggi yang populer digunakan oleh jutaan developer.

## 🎯 Learning Objectives
- Memahami sintaks dasar Python
- Mampu menulis program sederhana
- Mengenal tipe data Python

## 📖 Penjelasan Konsep

Python dirancang dengan filosofi keterbacaan kode. Sintaksnya yang bersih membuatnya
ideal untuk pemula maupun ahli. Python mendukung paradigma pemrograman prosedural,
berorientasi objek, dan fungsional.

### Variabel dan Tipe Data
```python
name = "Alice"       # str
age = 25             # int
gpa = 3.75           # float
is_student = True    # bool
```

## 💡 Contoh Konkret

Program "Hello, World!" paling sederhana di Python:
```python
print("Hello, World!")
```

## 🔁 Ringkasan
- Python adalah bahasa interpreted dan dynamically typed
- Mendukung multiple paradigma pemrograman
- Ekosistem library yang sangat kaya (NumPy, Pandas, Django, FastAPI)

## 🧪 Latihan Mandiri
1. Buat program yang menampilkan nama dan umur Anda.
2. Hitung luas lingkaran dengan jari-jari 7 menggunakan Python.
"""

async def test_indexing():
    print("=" * 60)
    print("TEST: RAG Indexing Pipeline")
    print("=" * 60)

    # Test 1: Chunking
    print("\n[1] Testing chunker...")
    from app.rag.chunker import chunk_markdown
    chunks = chunk_markdown(SAMPLE_MARKDOWN)
    print(f"    ✓ Produced {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        print(f"    Chunk {i+1}: {c[:80]}...")

    # Test 2: Embedding
    print("\n[2] Testing embedder (nomic-embed-text via Ollama)...")
    try:
        from app.rag.embedder import embed_texts, embed_query
        # Just embed a few chunks to be fast
        sample_chunks = chunks[:3]
        embeddings = embed_texts(sample_chunks)
        print(f"    ✓ Generated {len(embeddings)} embeddings, dim={len(embeddings[0])}")

        query_vec = embed_query("Apa itu Python?")
        print(f"    ✓ Query embedding dim={len(query_vec)}")
    except Exception as e:
        print(f"    ✗ Embedding failed: {e}")
        print("      → Make sure Ollama is running and nomic-embed-text is pulled")
        return

    # Test 3: Full indexing pipeline
    print("\n[3] Testing full index_module pipeline...")
    try:
        from app.rag.indexer import index_module
        count = await index_module(
            user_id="test_user_123",
            session_id="test_session_456",
            topic_id="python_intro",
            module_id="python_intro",
            title="Introduction to Python",
            content_markdown=SAMPLE_MARKDOWN,
            week=1,
            day=1,
        )
        print(f"    ✓ Indexed {count} chunks into Qdrant")
    except Exception as e:
        print(f"    ✗ Indexing failed: {e}")
        print("      → Make sure Qdrant is running on localhost:6333")
        return

    # Test 4: Search/retrieval from Qdrant
    print("\n[4] Testing Qdrant search...")
    try:
        from app.rag.vector_store import get_qdrant_client, search_chunks
        client = get_qdrant_client()
        results = search_chunks(
            client=client,
            user_id="test_user_123",
            query_vector=query_vec,
            topic_id="python_intro",
            top_k=3,
        )
        print(f"    ✓ Found {len(results)} matching chunks")
        for r in results:
            print(f"      Score={r['score']:.3f}: {r['text'][:60]}...")
    except Exception as e:
        print(f"    ✗ Search failed: {e}")

    print("\n" + "=" * 60)
    print("INDEXING TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_indexing())
