# currency_agent/__main__.py
import os
import uvicorn
import asyncio
from fastapi import FastAPI
from dotenv import load_dotenv

from .currency_agent_executor import CurrencyAgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

from .mcp_server import app as mcp_app

load_dotenv()

def build_app():
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 2000))
    mcp_port = int(os.getenv("MCP_PORT", 8001))

    # Validate required environment variables
    if not os.getenv("LOCAL_LLM_API_KEY"):
        raise ValueError("LOCAL_LLM_API_KEY environment variable must be set")

    if not os.getenv("EXCHANGE_API_KEY"):
        print("Warning: EXCHANGE_API_KEY not set. Using free tier with limitations.")

    # Define agent skill with more comprehensive examples
    skill = AgentSkill(
        id="currency_conversion",
        name="Currency Conversion",
        description="Convert currencies and provide exchange rate information using real-time data",
        tags=["currency", "finance", "exchange", "mcp"],
        examples=[
            "Convert 100 USD to EUR",
            "What is the exchange rate for GBP to JPY?",
            "How much is 50 euros in dollars?",
            "Show me rates for USD to multiple currencies",
            "What's the current rate from CAD to AUD?",
            "Convert 1000 JPY to USD"
        ],
    )

    # Create agent card
    agent_card = AgentCard(
        name="Currency Agent",
        description="I provide real-time currency conversion and exchange rate information using MCP tools.",
        url=f"http://{host}:{port}/agent/",
        # url=f""http://localhost:2000/agent/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        authentication={"schemes": ["public"]},
    )

    # Create agent executor with MCP integration
    mcp_base_url = f"http://localhost:{mcp_port}"
    agent_executor = CurrencyAgentExecutor(mcp_base_url=mcp_base_url)
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, 
        task_store=InMemoryTaskStore()
    )

    # Create A2A application
    # a2a_app = A2AStarletteApplication(agent_card, request_handler)
    a2a_app = A2AStarletteApplication(agent_card, request_handler).build()
    # Mount the A2A app on the MCP server
    mcp_app.mount("/agent", a2a_app)
    
    return mcp_app

async def start_services():
    """Start both MCP server and A2A agent"""
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 2000))
    
    app = build_app()
    
    # Start the combined server
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    
    print(f"🚀 Starting Currency Agent with MCP integration")
    print(f"📍 Agent endpoint: http://{host}:{port}/agent")
    print(f"🔧 MCP tools endpoint: http://{host}:{port}/tools/list")
    print(f"💱 Currency conversion ready!")
    
    await server.serve()

if __name__ == "__main__":
    asyncio.run(start_services())