# import asyncio
# import logging
# import os
# from collections.abc import AsyncIterable
# from typing import Any, Dict
# from uuid import uuid4

# import httpx
# from pydantic import ConfigDict
# from langchain_core.messages import HumanMessage, SystemMessage

# from local_llm_wrapper import LocalLLMChat

# from a2a.client import A2AClient
# from a2a.server.agent_execution import AgentExecutor, RequestContext
# from a2a.server.events.event_queue import EventQueue
# from a2a.server.tasks import TaskUpdater
# from a2a.types import (
#     Artifact,
#     FilePart,
#     FileWithBytes,
#     FileWithUri,
#     Message,
#     MessageSendParams,
#     Part,
#     Role,
#     SendMessageRequest,
#     SendMessageSuccessResponse,
#     Task,
#     # taskId,
#     TaskState,
#     TaskStatus,
#     TextPart,
#     UnsupportedOperationError,
# )
# from a2a.utils import get_text_parts
# from a2a.utils.errors import ServerError

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# TASK_POLLING_DELAY_SECONDS = 0.2

# class ClientAgentExecutor(AgentExecutor):
#     """Client Agent that routes queries to specialized agents using local LLM."""

#     def __init__(self, available_agents: Dict[str, Dict[str, Any]]):
#         self.available_agents = available_agents

#         # Build instruction with agent information
#         agent_descriptions = []
#         for agent_name, agent_info in available_agents.items():
#             keywords = ", ".join(agent_info['keywords'])
#             agent_descriptions.append(
#                 f"- {agent_name.title()} Agent: {agent_info['description']} (Keywords: {keywords})"
#             )

#         self.instruction = f"""
#         You are a Client Agent that routes user queries to specialized agents.

#         Available Agents:
#         {chr(10).join(agent_descriptions)}
        
#         Your job is to:
#         1. Analyze the user's query to understand what they need
#         2. Determine which specialized agent can best handle the request
#         3. Route the query to the appropriate agent using the available tools
#         4. Provide the response back to the user

#         If a query involves multiple domains (e.g., "What's the weather in Japan and convert 100 USD to JPY"), 
#         you can call multiple agents sequentially.

#         If the query is not related to any available agents, politely explain what you can help with.
#         """

#         self.llm = LocalLLMChat(
#             model=os.getenv("LOCAL_LLM_MODEL", "llama3.3"),
#             api_key=os.getenv("LOCAL_LLM_API_KEY")
#         )

#     async def execute(self, context: RequestContext, event_queue: EventQueue):
#         updater = TaskUpdater(event_queue, context.task_id, context.context_id)

#         if not context.current_task:
#             updater.submit()
#         updater.start_work()

#         # Extract the message text
#         user_message = ""
#         for part in context.message.parts:
#             if isinstance(part.root, TextPart):
#                 user_message += part.root.text

#         updater.update_status(
#             TaskState.working,
#             message=updater.new_agent_message(
#                 [Part(root=TextPart(text="Analyzing query with LLM..."))]
#             ),
#         )

#         # Route the query
#         result = await self._route_query(user_message, context, updater)

#         updater.add_artifact([Part(root=TextPart(text=result['response']))])
#         updater.complete()

#     async def _route_query(self, user_query: str, context: RequestContext, updater: TaskUpdater) -> Dict:
#         messages = [
#             SystemMessage(content=self.instruction),
#             HumanMessage(content=user_query)
#         ]
#         response_msg = self.llm.invoke(messages)
#         response_text = response_msg.content.lower()

#         logger.debug(f"LLM response for routing: {response_text}")
        
#         if any(k in response_text for k in ["weather", "temperature", "rain", "forecast", "climate"]):
#             logger.info("Routing to weather agent")
#             return await self.call_weather_agent(user_query, context)
#         elif any(k in response_text for k in ["currency", "convert", "exchange", "dollar", "euro", "yen", "rate"]):
#             logger.info("Routing to currency agent")
#             return await self.call_currency_agent(user_query, context)
#         else:
#             return {"response": "I'm not sure how to help with that. I can assist with weather or currency conversions."}

#     async def call_weather_agent(self, query: str, context: RequestContext):
#         return await self._call_specialized_agent('weather', query, context)

#     async def call_currency_agent(self, query: str, context: RequestContext):
#         return await self._call_specialized_agent('currency', query, context)

    
#     async def _call_specialized_agent(self, agent_name: str, query: str, context: RequestContext):
#         agent_info = self.available_agents[agent_name]
#         agent_url = agent_info['url'].rstrip('/') + '/'  # normalize
#         full_url = agent_url  # we know the JSON-RPC endpoint is at the base URL

#         print(f"Calling {agent_name} agent at {full_url}")

#         request = SendMessageRequest(
#             params=MessageSendParams(
#                 message=Message(
#                     contextId=context.context_id,
#                     taskId=None,
#                     messageId=str(uuid4()),
#                     role=Role.user,
#                     parts=[Part(root=TextPart(text=query))],
#                 )
#             )
#         )

#         try:
#             async with httpx.AsyncClient(timeout=30.0) as client:
#                 agent_client = A2AClient(httpx_client=client, url=full_url)
#                 response = await agent_client.send_message(request)
#         except Exception as e:
#             return {"response": f"Error communicating with {agent_name} agent: {e}"}

#         # —— now parse out the text parts and return! ——
#         content_lines = []
#         if isinstance(response.root, SendMessageSuccessResponse):
#             res = response.root.result
#             # handle both Task and direct message cases
#             artifacts = getattr(res, "artifacts", None)
#             if artifacts:
#                 for art in artifacts:
#                     for part in art.parts:
#                         if hasattr(part.root, "text"):
#                             content_lines.append(part.root.text)
#             else:
#                 msg = getattr(res, "status", None)
#                 if msg:
#                     for part in msg.message.parts:
#                         content_lines.append(part.root.text)
#                 else:
#                     # fallback on direct parts
#                     for part in getattr(res, "parts", []):
#                         content_lines.append(part.root.text)
#         else:
#             # non-success response
#             content_lines.append(str(response.root))

#         return {"response": "\n".join(content_lines) or f"No response from {agent_name} agent"}

#     async def cancel(self, context: RequestContext, event_queue: EventQueue):
#         raise ServerError(error=UnsupportedOperationError())

import asyncio
import logging
import os
from collections.abc import AsyncIterable
from typing import Any, Dict, List
from uuid import uuid4

import httpx
from pydantic import ConfigDict
from langchain_core.messages import HumanMessage, SystemMessage

from local_llm_wrapper import LocalLLMChat

from a2a.client import A2AClient
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import get_text_parts
from a2a.utils.errors import ServerError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TASK_POLLING_DELAY_SECONDS = 0.2

class ClientAgentExecutor(AgentExecutor):
    """Client Agent that routes queries to specialized agents using local LLM."""

    def __init__(self, available_agents: Dict[str, Dict[str, Any]]):
        self.available_agents = available_agents
        self._initialized = False
        self.agent_capabilities = {}
        
        self.llm = LocalLLMChat(
            model=os.getenv("LOCAL_LLM_MODEL", "llama3.3"),
            api_key=os.getenv("LOCAL_LLM_API_KEY")
        )

    async def _discover_agent_capabilities(self):
        """Dynamically discover what each agent can do by querying their MCP tools"""
        for agent_name, agent_info in self.available_agents.items():
            try:
                # Extract base URL and construct MCP tools endpoint
                base_url = agent_info['url'].rstrip('/').replace('/agent', '')
                tools_url = f"{base_url}/tools/list"
                
                logger.info(f"Discovering tools for {agent_name} agent at {tools_url}")
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(tools_url)
                    if response.status_code == 200:
                        tools = response.json()
                        self.agent_capabilities[agent_name] = {
                            'tools': tools,
                            'tool_names': [tool['name'] for tool in tools],
                            'keywords': set()  # Build keywords from tool descriptions
                        }
                        
                        # Extract keywords from tool names and descriptions
                        for tool in tools:
                            # Add tool name as keyword
                            self.agent_capabilities[agent_name]['keywords'].add(tool['name'].lower())
                            
                            # Extract keywords from description
                            if 'description' in tool:
                                desc_words = tool['description'].lower().split()
                                for word in ['weather', 'temperature', 'forecast', 'currency', 'convert', 'exchange', 'rate']:
                                    if word in desc_words:
                                        self.agent_capabilities[agent_name]['keywords'].add(word)
                        
                        logger.info(f"Discovered {len(tools)} tools for {agent_name}: {self.agent_capabilities[agent_name]['tool_names']}")
                    else:
                        logger.warning(f"Could not discover tools for {agent_name}: {response.status_code}")
                        self.agent_capabilities[agent_name] = {
                            'tools': [],
                            'tool_names': [],
                            'keywords': set(agent_info.get('keywords', []))
                        }
            except Exception as e:
                logger.error(f"Error discovering tools for {agent_name}: {e}")
                # Fallback to static keywords if discovery fails
                self.agent_capabilities[agent_name] = {
                    'tools': [],
                    'tool_names': [],
                    'keywords': set(agent_info.get('keywords', []))
                }

    async def _build_dynamic_instruction(self):
        """Build instruction prompt based on discovered capabilities"""
        agent_descriptions = []
        for agent_name, agent_info in self.available_agents.items():
            capabilities = self.agent_capabilities.get(agent_name, {})
            tools = capabilities.get('tool_names', [])
            keywords = capabilities.get('keywords', set())
            
            # Combine static description with discovered tools
            tools_str = ", ".join(tools) if tools else "No tools discovered"
            keywords_str = ", ".join(keywords) if keywords else agent_info.get('keywords', [])
            
            agent_descriptions.append(
                f"- {agent_name.title()} Agent: {agent_info['description']}\n"
                f"  Available tools: {tools_str}\n"
                f"  Keywords: {keywords_str}"
            )

        self.instruction = f"""
        You are a Client Agent that routes user queries to specialized agents.

        Available Agents and their capabilities:
        {chr(10).join(agent_descriptions)}
        
        Your job is to:
        1. Analyze the user's query to understand what they need
        2. Determine which specialized agent can best handle the request based on their available tools
        3. Route the query to the appropriate agent
        4. Provide the response back to the user

        If a query involves multiple domains, you can call multiple agents sequentially.
        If the query is not related to any available agents' tools, politely explain what you can help with.
        """

    async def initialize(self):
        """Initialize the client agent by discovering available tools"""
        if self._initialized:
            return
            
        logger.info("Initializing client agent with dynamic tool discovery...")
        await self._discover_agent_capabilities()
        await self._build_dynamic_instruction()
        self._initialized = True
        logger.info("Client agent initialization complete")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Initialize if not already done
        if not self._initialized:
            await self.initialize()
            
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if not context.current_task:
            updater.submit()
        updater.start_work()

        # Extract the message text
        user_message = ""
        for part in context.message.parts:
            if isinstance(part.root, TextPart):
                user_message += part.root.text

        updater.update_status(
            TaskState.working,
            message=updater.new_agent_message(
                [Part(root=TextPart(text="Analyzing query and routing to appropriate agent..."))]
            ),
        )

        # Route the query with dynamic instruction
        result = await self._route_query(user_message, context, updater)

        updater.add_artifact([Part(root=TextPart(text=result['response']))])
        updater.complete()

    async def _route_query(self, user_query: str, context: RequestContext, updater: TaskUpdater) -> Dict:
        messages = [
            SystemMessage(content=self.instruction),
            HumanMessage(content=user_query)
        ]
        response_msg = self.llm.invoke(messages)
        response_text = response_msg.content.lower()

        logger.debug(f"LLM response for routing: {response_text}")
        
        # Check against discovered capabilities
        for agent_name, capabilities in self.agent_capabilities.items():
            keywords = capabilities.get('keywords', set())
            tool_names = capabilities.get('tool_names', [])
            
            # Check if response mentions any tools or keywords
            if any(tool.lower() in response_text for tool in tool_names) or \
               any(keyword in response_text for keyword in keywords):
                logger.info(f"Routing to {agent_name} agent based on discovered capabilities")
                return await self._call_specialized_agent(agent_name, user_query, context)
        
        # Fallback to keyword matching if no specific tool mentioned
        if any(k in response_text for k in ["weather", "temperature", "rain", "forecast", "climate"]):
            logger.info("Routing to weather agent")
            return await self._call_specialized_agent('weather', user_query, context)
        elif any(k in response_text for k in ["currency", "convert", "exchange", "dollar", "euro", "yen", "rate"]):
            logger.info("Routing to currency agent")
            return await self._call_specialized_agent('currency', user_query, context)
        else:
            # Build a helpful message about available capabilities
            available_tools = []
            for agent_name, caps in self.agent_capabilities.items():
                if caps['tool_names']:
                    available_tools.extend([f"{agent_name}: {tool}" for tool in caps['tool_names']])
            
            tools_msg = "\n".join(available_tools) if available_tools else "No tools currently available"
            return {
                "response": f"I'm not sure how to help with that. I can assist with the following:\n\n{tools_msg}"
            }

    async def _call_specialized_agent(self, agent_name: str, query: str, context: RequestContext):
        agent_info = self.available_agents[agent_name]
        agent_url = agent_info['url'].rstrip('/') + '/'
        full_url = agent_url

        print(f"Calling {agent_name} agent at {full_url}")

        request = SendMessageRequest(
            params=MessageSendParams(
                message=Message(
                    contextId=context.context_id,
                    taskId=None,
                    messageId=str(uuid4()),
                    role=Role.user,
                    parts=[Part(root=TextPart(text=query))],
                )
            )
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                agent_client = A2AClient(httpx_client=client, url=full_url)
                response = await agent_client.send_message(request)
        except Exception as e:
            return {"response": f"Error communicating with {agent_name} agent: {e}"}

        # Parse response (same as before)
        content_lines = []
        if isinstance(response.root, SendMessageSuccessResponse):
            res = response.root.result
            artifacts = getattr(res, "artifacts", None)
            if artifacts:
                for art in artifacts:
                    for part in art.parts:
                        if hasattr(part.root, "text"):
                            content_lines.append(part.root.text)
            else:
                msg = getattr(res, "status", None)
                if msg:
                    for part in msg.message.parts:
                        content_lines.append(part.root.text)
                else:
                    for part in getattr(res, "parts", []):
                        content_lines.append(part.root.text)
        else:
            content_lines.append(str(response.root))

        return {"response": "\n".join(content_lines) or f"No response from {agent_name} agent"}

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())