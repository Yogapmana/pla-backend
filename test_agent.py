import asyncio
from app.agents.general_chatbot import build_agent_tools, get_llm, SYSTEM_PROMPT, create_react_agent

def test():
    user_id = "test-user"
    session_id = "test-session"
    tools = build_agent_tools(user_id, session_id)
    llm = get_llm("gemma-7b") # dummy model
    agent = create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)
    print("Agent created successfully!")

if __name__ == "__main__":
    test()
