# weather_agent/__main__.py
import os
import uvicorn
import asyncio
from fastapi import FastAPI
from dotenv import load_dotenv

from .weather_agent_executor import WeatherAgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

from .mcp_server import app as mcp_app

load_dotenv()

def build_app():
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 3000))
    mcp_port = int(os.getenv("MCP_PORT", 8002))

    # Validate required environment variables
    if not os.getenv("LOCAL_LLM_API_KEY"):
        raise ValueError("LOCAL_LLM_API_KEY environment variable must be set")

    if not os.getenv("WEATHER_API_KEY"):
        print("Warning: WEATHER_API_KEY not set. Using free tier with limitations.")

    # Define agent skill with comprehensive examples
    skill = AgentSkill(
        id="weather_info",
        name="Weather Information",
        description="Provide weather information, forecasts, and climate data for any location using real-time data",
        tags=["weather", "forecast", "climate", "mcp"],
        examples=[
            "What is the weather like in New York?",
            "Will it rain tomorrow in London?",
            "What is the temperature in Tokyo?",
            "Give me the 5-day forecast for San Francisco",
            "How's the weather in Paris today?",
            "What's the current temperature in Mumbai?"
        ],
    )

    # Create agent card
    agent_card = AgentCard(
        name="Weather Agent",
        description="I provide accurate weather information and forecasts for any location worldwide using MCP tools.",
        url=f"http://{host}:{port}/agent/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        authentication={"schemes": ["public"]},
    )

    # Create agent executor with MCP integration
    mcp_base_url = f"http://localhost:{mcp_port}"
    agent_executor = WeatherAgentExecutor(mcp_base_url=mcp_base_url)
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, 
        task_store=InMemoryTaskStore()
    )

    # Create A2A application and mount on MCP server
    a2a_app = A2AStarletteApplication(agent_card, request_handler).build()
    mcp_app.mount("/agent", a2a_app)
    
    return mcp_app

async def start_services():
    """Start both MCP server and A2A agent"""
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 3000))
    
    app = build_app()
    
    # Start the combined server
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    
    print(f"🚀 Starting Weather Agent with MCP integration")
    print(f"📍 Agent endpoint: http://{host}:{port}/agent")
    print(f"🔧 MCP tools endpoint: http://{host}:{port}/tools/list")
    print(f"🌤️ Weather information ready!")
    
    await server.serve()

if __name__ == "__main__":
    asyncio.run(start_services())