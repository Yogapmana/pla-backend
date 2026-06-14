import asyncio
import json
import uuid
from app.agents.orchestrator import build_pla_graph
from app.agents.state import LearningConfig

async def main():
    print("Initializing test PLA session...")
    
    # Initialize basic state
    initial_state = {
        "user_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "learning_config": LearningConfig(
            topic="Basic Python Programming",
            duration_weeks=1,
            level="beginner",
            hours_per_day=1.0,
            language="id"
        ),
        "agent_logs": [],
        "curriculum": None,
        "research_results": [],
        "modules": [],
        "chat_history": [],
        "quiz_results": [],
        "mastery_scores": {},
        "progress_signals": None,
        "feedback_actions": []
    }
    
    print("Starting LangGraph execution...")
    # Because researcher is async and invokes async tools, we use ainvoke
    graph = build_pla_graph().compile()
    final_state = await graph.ainvoke(initial_state)
    
    print("\n" + "="*50)
    print("EXECUTION COMPLETED")
    print("="*50)
    
    print("\n--- AGENT LOGS ---")
    for log in final_state.get("agent_logs", []):
        print(f"[{log.timestamp.strftime('%H:%M:%S')}] [{log.agent.upper()}] {log.message}")
        
    print("\n--- CURRICULUM ---")
    curriculum = final_state.get("curriculum")
    if curriculum:
        print(f"Topic: {curriculum.topic}")
        print(f"Total Weeks: {curriculum.total_weeks}")
        for week in curriculum.weeks:
            print(f"  Week {week.week}: {week.title}")
            for day in week.days:
                print(f"    Day {day.day}: {day.title} (Status: {day.status})")
                print(f"      Queries: {day.search_queries}")
    else:
        print("No curriculum generated.")
        
    print("\n--- RESEARCH RESULTS ---")
    results = final_state.get("research_results", [])
    print(f"Total sources found: {len(results)}")
    for i, r in enumerate(results[:3]):
        print(f"  {i+1}. [{r.source_type}] {r.source_title}")
    if len(results) > 3:
        print("  ...")
        
    print("\n--- COMPOSER MODULES ---")
    modules = final_state.get("modules", [])
    print(f"Total modules generated: {len(modules)}")
    for m in modules:
        print(f"\n[Module for Topic: {m.topic_id} | Title: {m.title}]")
        # Print just the first 300 chars of markdown
        snippet = m.content_markdown[:300].replace('\n', ' ')
        print(f"Snippet: {snippet}...")
        print(f"Sources used: {len(m.sources)}")

if __name__ == "__main__":
    asyncio.run(main())
