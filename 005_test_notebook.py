import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents.middleware import TodoListMiddleware
import subprocess, sys

@tool
def finance_researcher(query: str):
    """Research stocks using Yahoo Finance MCP async function."""
    code = f"""
import asyncio
from scripts.yahoo_mcp import finance_research
asyncio.run(finance_research("{query}"))
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True,
        cwd=r'D:\coding\ai_learning\laxmimerit\Multi-Agent-Deep-RAG'
    )
    if result.returncode != 0:
        print('STDERR:', result.stderr[:2000])
    return result.stdout

system_prompt = "You are a professional stock research analyst."
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("Creating agent...")
try:
    agent = create_agent(
        model=model,
        tools=[finance_researcher],
        system_prompt=system_prompt,
        middleware=[TodoListMiddleware()]
    )
    print("Agent created OK:", type(agent))
except Exception as e:
    print(f"ERROR creating agent: {type(e).__name__}: {e}")

print("\nInvoking agent with non-finance question...")
try:
    response = agent.invoke({'messages': [HumanMessage('what is the weather in mumbai?')]})
    print("Response OK")
    last_msg = response['messages'][-1]
    print("Last message:", last_msg.text if hasattr(last_msg, 'text') else last_msg.content)
except Exception as e:
    import traceback
    print(f"ERROR invoking agent: {type(e).__name__}: {e}")
    traceback.print_exc()
