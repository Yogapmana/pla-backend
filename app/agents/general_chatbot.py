import time
import logging
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.utils.llm_factory import get_llm
from app.tools.tavily_search import tavily_search_tool as _tavily_tool
from app.tools.wikipedia_search import wikipedia_search_tool as _wiki_tool
from app.tools.youtube_transcript import youtube_transcript_tool as _youtube_tool
from app.tools.jina_reader import jina_read_urls
from app.rag.retriever import retrieve_and_rerank

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Anda adalah Asisten AI Pembelajaran yang canggih dan mandiri (Independent Personal Learning Agent).
Tugas Anda adalah membantu pengguna belajar, menjawab pertanyaan teknis maupun umum, serta mendiskusikan materi.

Anda dilengkapi dengan berbagai alat (tools) yang dapat digunakan jika diperlukan:
1. Pencarian Web (Tavily): Gunakan untuk mencari informasi terkini di internet.
2. Wikipedia: Gunakan untuk mencari definisi, fakta sejarah, atau konsep ensiklopedik.
3. YouTube Transcript: Gunakan jika pengguna memberikan tautan video YouTube untuk membaca transkripnya.
4. Scrape Webpage: Gunakan untuk mengekstrak isi teks penuh dari sebuah tautan (URL).
5. Search My Documents: Gunakan UTAMANYA JIKA pengguna bertanya tentang "materi saya", "dokumen saya", "kurikulum ini", "apa yang sudah saya pelajari", atau konteks lokal sesi mereka.

ATURAN PENTING:
- Gunakan bahasa Indonesia yang natural, ramah, dan edukatif.
- Jangan menebak-nebak fakta. Jika Anda tidak tahu atau jika informasi berkaitan dengan peristiwa terkini, GUNAKAN tool pencarian.
- JIKA Anda menggunakan sumber dari web atau dokumen, selalu cantumkan referensinya dalam jawaban (misal dengan Markdown link).
- Jika pengguna hanya mengobrol biasa (say hello), Anda tidak perlu memanggil tool. Jawablah secara langsung.
"""

def build_agent_tools(user_id: str, session_id: str):
    """Build a dynamic toolset that injects user_id and session_id into the RAG tool."""
    
    from app.utils.log_broker import publish_log
    import asyncio

    @tool
    async def scrape_webpage(url: str) -> str:
        """Scrape full text content from a given web page URL. Use this to read articles or documentation."""
        await publish_log(session_id, {"type": "tool_start", "agent": "chatbot", "level": "info", "message": f"Membaca isi halaman web: {url[:50]}..."})
        try:
            results = await jina_read_urls([url], timeout=20)
            if results and results[0].get("success"):
                text = results[0]["text"]
                return text[:4000] + ("..." if len(text) > 4000 else "")
            return f"Gagal membaca URL: {results[0].get('error', 'Unknown error') if results else 'Empty'}"
        except Exception as e:
            return f"Error scraping URL: {str(e)}"
    
    @tool
    async def search_my_documents(query: str) -> str:
        """Search the user's uploaded documents and learning curriculum materials for the given query.
        Use this tool when the user asks about their specific study materials, PDFs, or past topics in this session.
        """
        await publish_log(session_id, {"type": "tool_start", "agent": "chatbot", "level": "info", "message": f"Mencari dokumen lokal: {query}"})
        try:
            # retrieve_and_rerank is synchronous, run in thread to avoid blocking
            chunks = await asyncio.to_thread(
                retrieve_and_rerank,
                user_id=user_id,
                query=query,
                session_id=session_id,
                topic_id=None,
                top_k_retrieve=8,
                top_k_rerank=4,
                use_hyde=True
            )
            if not chunks:
                return "Tidak ada dokumen relevan yang ditemukan dalam materi Anda."
            
            blocks = []
            for i, chunk in enumerate(chunks, 1):
                source = chunk.get("source_title", "Dokumen Tidak Bernama")
                text = chunk.get("text", "")
                blocks.append(f"[Sumber {i}: {source}]\n{text}")
            return "\n\n---\n\n".join(blocks)
        except Exception as e:
            logger.error(f"[GeneralChatbot] RAG error: {e}")
            return f"Terjadi kesalahan saat mencari dokumen: {str(e)}"

    @tool
    async def tavily_search(query: str) -> str:
        """Search the web for information using Tavily API. 
        Useful for finding up-to-date information, articles, and general knowledge."""
        await publish_log(session_id, {"type": "tool_start", "agent": "chatbot", "level": "info", "message": f"Mencari web: {query}"})
        try:
            results = await asyncio.to_thread(_tavily_tool.invoke, {"query": query})
            if not results:
                return "Tidak ada hasil ditemukan."
            blocks = []
            for i, r in enumerate(results, 1):
                raw = str(r.get('raw_text', ''))
                blocks.append(f"[{i}] {r.get('source_title', '')}\nURL: {r.get('source_url', '')}\n{raw[:800]}...")
            return "\n\n".join(blocks)[:4000]
        except Exception as e:
            return f"Error: {e}"

    @tool
    async def wikipedia_search(query: str) -> str:
        """Search Wikipedia for a given topic and return the summary.
        Useful for factual information, definitions, and broad topic overviews."""
        await publish_log(session_id, {"type": "tool_start", "agent": "chatbot", "level": "info", "message": f"Mencari Wikipedia: {query}"})
        try:
            results = await asyncio.to_thread(_wiki_tool.invoke, {"query": query})
            if not results:
                return "Tidak ada hasil ditemukan di Wikipedia."
            blocks = []
            for i, r in enumerate(results, 1):
                raw = str(r.get('summary', ''))
                blocks.append(f"[{i}] {r.get('title', '')}\nURL: {r.get('url', '')}\n{raw[:1000]}...")
            return "\n\n".join(blocks)[:2000]
        except Exception as e:
            return f"Error: {e}"

    @tool
    async def youtube_transcript(url: str) -> str:
        """Extract transcript from a YouTube video URL."""
        await publish_log(session_id, {"type": "tool_start", "agent": "chatbot", "level": "info", "message": f"Membaca transkrip YouTube: {url}"})
        try:
            results = await asyncio.to_thread(_youtube_tool.invoke, {"url": url})
            if not results:
                return "Tidak ada transkrip yang ditemukan untuk video tersebut."
            blocks = []
            for i, r in enumerate(results, 1):
                blocks.append(f"[{i}] {r.get('title', '')} ({r.get('author', '')})\n{str(r.get('transcript', ''))[:3000]}...")
            return "\n\n".join(blocks)[:3000]
        except Exception as e:
            return f"Error: {e}"

    return [
        tavily_search,
        wikipedia_search,
        youtube_transcript,
        scrape_webpage,
        search_my_documents
    ]

async def general_chatbot_chat(
    user_id: str,
    session_id: str,
    query: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Independent ReAct Agent Chat Mode:
    1. Sets up tools dynamically for the current session.
    2. Uses langgraph create_react_agent to reason and act.
    3. Returns the final response.
    """
    start_time = time.time()
    chat_history = chat_history or []
    
    logger.info(f"[GeneralChatbot] Processing query for session '{session_id}': {query[:80]}...")

    llm = get_llm(settings.GENERAL_CHAT_MODEL)
    tools = build_agent_tools(user_id, session_id)
    
    # Add current date to system prompt dynamically
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dynamic_system_prompt = f"{SYSTEM_PROMPT}\n\nWaktu saat ini: {current_date}\nIngat bahwa Anda harus menggunakan Web Search jika ditanya tentang peristiwa yang terjadi dekat dengan atau setelah tanggal ini!"
    
    # create_react_agent handles the tool loop internally.
    agent = create_react_agent(llm, tools=tools, prompt=dynamic_system_prompt)
    
    # Format chat history into LangChain messages
    messages = []
    # Only keep the last 6 messages to avoid hitting token limits
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in recent_history:
        role = msg.get("role")
        content = msg.get("content", "")
        # Truncate past messages to save context space
        if len(content) > 1000:
            content = content[:1000] + " ...[truncated]"
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
            
    # Add the current query
    messages.append(HumanMessage(content=query))
    
    try:
        # We must use ainvoke for async tools to work correctly within the react loop
        result = await agent.ainvoke({"messages": messages})
        
        # The result state contains the list of all messages. The last one is the final AI response.
        final_message = result["messages"][-1].content
        
        # Check if the agent used any tools
        # We can extract tools used by inspecting the messages list
        tools_used = sum(1 for m in result["messages"] if m.type == "tool")
    except Exception as e:
        logger.error(f"[GeneralChatbot] Error during agent execution: {e}")
        final_message = "Maaf, saya mengalami kesalahan saat memproses permintaan Anda. Silakan coba lagi."
        tools_used = 0

    latency_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"[GeneralChatbot] Response generated in {latency_ms}ms using {tools_used} tool calls."
    )

    # General chatbot doesn't strictly track chunks for Ragas in the same way Tutor does,
    # as it might not use RAG at all. We pass an empty list of sources, relying on AI to mention them.
    return {
        "response": final_message,
        "sources": [],
        "latency_ms": latency_ms,
        "chunks_used": tools_used, # Hijacking this field to track tool calls
        "timestamp": datetime.utcnow().isoformat(),
    }
