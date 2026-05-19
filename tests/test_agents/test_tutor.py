"""
Test Tutor Agent — Chat RAG Mode

Requires:
- nomic-embed-text indexed data in Qdrant (run test_indexer.py first)
- TUTOR_MODEL accessible (Groq or Ollama)

Run with: PYTHONPATH=. ./venv/bin/python tests/test_agents/test_tutor.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

async def test_tutor_chat():
    print("=" * 60)
    print("TEST: Tutor Agent — Chat RAG Mode")
    print("=" * 60)

    print("\n[PREREQ] Indexing sample module into Qdrant first...")
    try:
        from app.rag.indexer import index_module
        count = await index_module(
            user_id="test_user_tutor",
            session_id="session_tutor_001",
            topic_id="python_intro",
            module_id="python_intro",
            title="Introduction to Python",
            content_markdown="""
# Introduction to Python

Python adalah bahasa pemrograman yang populer. Diciptakan oleh Guido van Rossum pada 1991.

## Tipe Data
Python memiliki beberapa tipe data dasar: int, float, str, bool, list, dict, tuple.

## Fungsi
Fungsi di Python didefinisikan dengan keyword `def`:
```python
def greet(name):
    return f"Hello, {name}!"
```

## Kontrol Alur
Python menggunakan indentasi untuk blok kode:
```python
if x > 0:
    print("positif")
elif x < 0:
    print("negatif")
else:
    print("nol")
```
            """,
            week=1,
            day=1,
        )
        print(f"    ✓ Indexed {count} chunks")
    except Exception as e:
        print(f"    ✗ Prereq failed: {e}")
        return

    print("\n[1] Testing Tutor Chat RAG...")
    try:
        from app.agents.tutor import tutor_chat
        query = "Apa itu fungsi di Python dan bagaimana cara membuatnya?"
        result = await tutor_chat(
            user_id="test_user_tutor",
            session_id="session_tutor_001",
            topic_id="python_intro",
            query=query,
            include_sources=True,
        )
        print(f"    ✓ Response received in {result['latency_ms']}ms")
        print(f"    Chunks used: {result['chunks_used']}")
        print(f"    Sources: {[s['title'] for s in result.get('sources', [])]}")
        print(f"\n--- TUTOR RESPONSE ---")
        print(result["response"][:500])
        print("---")
    except Exception as e:
        print(f"    ✗ Tutor chat failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[2] Testing Quiz Generation...")
    try:
        from app.agents.tutor import tutor_generate_quiz
        questions = await tutor_generate_quiz(
            user_id="test_user_tutor",
            topic_id="python_intro",
            topic_title="Introduction to Python",
            num_questions=3,
        )
        print(f"    ✓ Generated {len(questions)} quiz questions")
        for i, q in enumerate(questions, 1):
            print(f"    Q{i}: {q.get('question', '')[:80]}")
    except Exception as e:
        print(f"    ✗ Quiz generation failed: {e}")

    print("\n" + "=" * 60)
    print("TUTOR AGENT TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_tutor_chat())
