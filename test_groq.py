import asyncio
from app.config import settings
from app.utils.llm_factory import get_llm
from app.agents.state import Curriculum
from langchain_core.prompts import ChatPromptTemplate

async def main():
    llm = get_llm("llama-3.3-70b-versatile", temperature=0.2)
    structured_llm = llm.with_structured_output(Curriculum)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Anda adalah AI Planner. KEMBALIKAN OUTPUT DALAM FORMAT JSON SESUAI DENGAN SKEMA YANG DIMINTA."),
        ("human", "Tolong buatkan kurikulum untuk saya tentang Python.")
    ])
    
    chain = prompt | structured_llm
    
    try:
        res = await chain.ainvoke({})
        print(res)
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, 'response'):
            print(e.response.text)

if __name__ == "__main__":
    asyncio.run(main())
