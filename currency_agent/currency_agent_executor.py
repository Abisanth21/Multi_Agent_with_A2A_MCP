
# # currency_agent/currency_agent_executor.py
# import asyncio
# import logging
# from typing import Dict, Any, List, Optional
# import json
# import os
# import httpx

# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
# from langchain_core.tools import tool, Tool
# from typing_extensions import Annotated, TypedDict

# from local_llm_wrapper import LocalLLMChat
# from a2a.server.agent_execution import AgentExecutor, RequestContext
# from a2a.server.events.event_queue import EventQueue
# from a2a.server.tasks import TaskUpdater
# from a2a.types import Part, TaskState, TextPart, UnsupportedOperationError
# from a2a.utils.errors import ServerError
# from pydantic import BaseModel

# class ConvertCurrencyArgs(BaseModel):
#     amount: float
#     from_currency: str
#     to_currency: str

# class GetExchangeRateArgs(BaseModel):
#     from_currency: str
#     to_currency: str

# class GetMultipleRatesArgs(BaseModel):
#     base_currency: str
#     target_currencies: Optional[List[str]] = None

# logger = logging.getLogger(__name__)

# class CurrencyState(TypedDict):
#     messages: Annotated[list, add_messages]

# class CurrencyAgentExecutor(AgentExecutor):
#     """Currency Agent using LangGraph with MCP integration."""

#     def __init__(self, mcp_base_url: str = "http://localhost:2000"):
#         self.llm = LocalLLMChat(
#             model=os.getenv("LOCAL_LLM_MODEL", "llama3.3"),
#             api_key=os.getenv("LOCAL_LLM_API_KEY")
#         )

#         self.mcp_base_url = mcp_base_url
#         self.graph = None
#         self._initialized = False

#     async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
#         """Call MCP tool with proper error handling and parameter validation"""
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             try:
#                 # Validate and clean arguments
#                 cleaned_args = {}
#                 if tool_name == "convertCurrency":
#                     cleaned_args = {
#                         "amount": float(arguments.get("amount", 0)),
#                         "from_currency": str(arguments.get("from_currency", "")).upper(),
#                         "to_currency": str(arguments.get("to_currency", "")).upper()
#                     }
#                 elif tool_name == "getExchangeRate":
#                     cleaned_args = {
#                         "from_currency": str(arguments.get("from_currency", "")).upper(),
#                         "to_currency": str(arguments.get("to_currency", "")).upper()
#                     }
#                 elif tool_name == "getMultipleRates":
#                     cleaned_args = {
#                         "base_currency": str(arguments.get("base_currency", "")).upper(),
#                         "target_currencies": arguments.get("target_currencies")
#                     }

#                 payload = {"name": tool_name, "arguments": cleaned_args}
#                 logger.info(f"Calling MCP tool {tool_name} with payload: {payload}")
                
#                 response = await client.post(
#                     f"{self.mcp_base_url}/tools/call",
#                     json=payload,
#                     headers={"Content-Type": "application/json"}
#                 )
#                 response.raise_for_status()
#                 result = response.json()
#                 logger.info(f"MCP tool {tool_name} response: {result}")
#                 return result
#             except httpx.HTTPStatusError as e:
#                 error_text = e.response.text
#                 logger.error(f"HTTP error calling MCP tool {tool_name}: {e.response.status_code} - {error_text}")
#                 return {"error": f"HTTP {e.response.status_code}: {error_text}"}
#             except Exception as e:
#                 logger.error(f"Failed to call MCP tool {tool_name}: {e}")
#                 return {"error": str(e)}

#     async def _get_available_tools(self) -> List[Dict[str, Any]]:
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             try:
#                 response = await client.get(f"{self.mcp_base_url}/tools/list")
#                 response.raise_for_status()
#                 return response.json()
#             except Exception as e:
#                 logger.error(f"Failed to fetch MCP tools: {e}")
#                 return []

#     def _create_currency_graph(self) -> StateGraph:

#         def should_continue(state: CurrencyState) -> str:
#             messages = state["messages"]
#             if not messages:
#                 return END
#             last_message = messages[-1]
#             if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#                 return "tools"
#             return END

#         async def call_model(state: CurrencyState):
#             messages = state["messages"]
#             system_message = SystemMessage(content="""
#             You are a helpful currency conversion agent. Your job is to provide accurate exchange rates and currency conversions using real-time data.

#             When users ask about currencies:
#             1. Extract the currencies and amounts from their query
#             2. Use appropriate tools to get real-time exchange rate data
#             3. Present the information in a clear, user-friendly format
#             4. For conversions, show both the rate and the converted amount

#             Available tools:
#             - convertCurrency: Convert a specific amount from one currency to another
#             - getExchangeRate: Get the current exchange rate between two currencies  
#             - getMultipleRates: Get rates from one base currency to multiple others

#             Common currency codes: USD, EUR, GBP, JPY, AUD, CAD, CHF, CNY, INR, etc.

#             Always use the tools to get real-time data. Present results in a friendly, conversational manner.
            
#             Example usage:
#             - For "convert 100 USD to EUR": use convertCurrency with amount=100, from_currency="USD", to_currency="EUR"
#             - For "what's the rate from GBP to JPY": use getExchangeRate with from_currency="GBP", to_currency="JPY"
#             """)
#             full_messages = [system_message] + messages

#             # Create proper tool definitions
#             def make_tool(name, description, args_schema):
#                 async def tool_wrapper(**kwargs):
#                     return f"Tool '{name}' would be called with: {kwargs}"
#                 return Tool.from_function(
#                     func=tool_wrapper,
#                     name=name,
#                     description=description,
#                     args_schema=args_schema
#                 )

#             tools = [
#                 make_tool("convertCurrency", 
#                          "Convert currency amount from one currency to another", 
#                          ConvertCurrencyArgs),
#                 make_tool("getExchangeRate", 
#                          "Get exchange rate between two currencies", 
#                          GetExchangeRateArgs),
#                 make_tool("getMultipleRates", 
#                          "Get multiple exchange rates from base currency", 
#                          GetMultipleRatesArgs)
#             ]

#             llm_with_tools = self.llm.bind_tools(tools)
#             response = llm_with_tools.invoke(full_messages)
#             return {"messages": [response]}

#         async def call_tools(state: CurrencyState):
#             messages = state["messages"]
#             if not messages:
#                 return {"messages": []}
#             last_message = messages[-1]
#             tool_messages = []

#             if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#                 for tool_call in last_message.tool_calls:
#                     tool_name = tool_call["name"]
#                     tool_args = tool_call["args"]
#                     tool_call_id = tool_call.get("id", "")
                    
#                     logger.info(f"Calling MCP tool: {tool_name} with args: {tool_args}")
#                     result = await self._call_mcp_tool(tool_name, tool_args)
                    
#                     # Format the result nicely
#                     if "error" in result:
#                         content = f"Error: {result['error']}"
#                     else:
#                         if tool_name == "convertCurrency":
#                             content = f"Conversion result: {result.get('result', 'N/A')} {result.get('to_currency', '')} (Rate: {result.get('rate', 'N/A')})"
#                         elif tool_name == "getExchangeRate":
#                             content = f"Exchange rate: 1 {result.get('from_currency', '')} = {result.get('rate', 'N/A')} {result.get('to_currency', '')}"
#                         elif tool_name == "getMultipleRates":
#                             rates = result.get('rates', {})
#                             rate_strings = [f"{curr}: {rate}" for curr, rate in rates.items()]
#                             content = f"Rates from {result.get('base_currency', '')}: {', '.join(rate_strings)}"
#                         else:
#                             content = json.dumps(result)
                    
#                     tool_message = ToolMessage(content=content, tool_call_id=tool_call_id)
#                     tool_messages.append(tool_message)
#             return {"messages": tool_messages}

#         workflow = StateGraph(CurrencyState)
#         workflow.add_node("agent", call_model)
#         workflow.add_node("tools", call_tools)
#         workflow.set_entry_point("agent")
#         workflow.add_conditional_edges("agent", should_continue)
#         workflow.add_edge("tools", "agent")
#         return workflow.compile()

#     async def initialize(self):
#         if self._initialized:
#             return
#         try:
#             tools = await self._get_available_tools()
#             logger.info(f"Found {len(tools)} MCP tools available")
#         except Exception as e:
#             logger.warning(f"Could not connect to MCP server: {e}")
#         self.graph = self._create_currency_graph()
#         self._initialized = True

#     async def execute(self, context: RequestContext, event_queue: EventQueue):
#         if not self._initialized:
#             await self.initialize()
#         updater = TaskUpdater(event_queue, context.task_id, context.context_id)
#         if not context.current_task:
#             updater.submit()
#         updater.start_work()

#         try:
#             user_message = ""
#             for part in context.message.parts:
#                 if isinstance(part.root, TextPart):
#                     user_message += part.root.text
#             if not user_message.strip():
#                 raise ValueError("No text content found in message")
#             logger.info(f"Processing currency request: {user_message}")
#             updater.update_status(
#                 TaskState.working,
#                 message=updater.new_agent_message([
#                     Part(root=TextPart(text="Processing your currency request..."))
#                 ]),
#             )
#             initial_state = CurrencyState(messages=[HumanMessage(content=user_message)])
#             final_state = await self.graph.ainvoke(initial_state)
#             if final_state["messages"]:
#                 final_message = final_state["messages"][-1]
#                 response_text = final_message.content
#             else:
#                 response_text = "I apologize, but I couldn't process your currency request."
#             logger.info(f"Currency agent response: {response_text}")
#             response_parts = [Part(root=TextPart(text=response_text))]
#             updater.add_artifact(response_parts)
#             updater.complete()

#         except Exception as e:
#             logger.error(f"Error in currency agent execution: {e}", exc_info=True)
#             error_message = f"I apologize, but I encountered an error while processing your currency request: {str(e)}"
#             error_parts = [Part(root=TextPart(text=error_message))]
#             updater.add_artifact(error_parts)
#             updater.complete()

#     async def cancel(self, context: RequestContext, event_queue: EventQueue):
#         raise ServerError(error=UnsupportedOperationError())

import asyncio
import logging
from typing import Dict, Any, List, Optional
import json
import os
import httpx

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, Tool
from typing_extensions import Annotated, TypedDict

from local_llm_wrapper import LocalLLMChat
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState, TextPart, UnsupportedOperationError
from a2a.utils.errors import ServerError
from pydantic import BaseModel, create_model
from typing import Type

logger = logging.getLogger(__name__)

class CurrencyState(TypedDict):
    messages: Annotated[list, add_messages]

class CurrencyAgentExecutor(AgentExecutor):
    """Currency Agent using LangGraph with dynamic MCP tool discovery."""

    def __init__(self, mcp_base_url: str = "http://localhost:2000"):
        self.llm = LocalLLMChat(
            model=os.getenv("LOCAL_LLM_MODEL", "llama3.3"),
            api_key=os.getenv("LOCAL_LLM_API_KEY")
        )

        self.mcp_base_url = mcp_base_url
        self.graph = None
        self._initialized = False
        self.available_tools = []
        self.tool_schemas = {}

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool with proper error handling"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                payload = {"name": tool_name, "arguments": arguments}
                logger.info(f"Calling MCP tool {tool_name} with payload: {payload}")
                
                response = await client.post(
                    f"{self.mcp_base_url}/tools/call",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"MCP tool {tool_name} response: {result}")
                return result
            except httpx.HTTPStatusError as e:
                error_text = e.response.text
                logger.error(f"HTTP error calling MCP tool {tool_name}: {e.response.status_code} - {error_text}")
                return {"error": f"HTTP {e.response.status_code}: {error_text}"}
            except Exception as e:
                logger.error(f"Failed to call MCP tool {tool_name}: {e}")
                return {"error": str(e)}

    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Fetch available tools from MCP server"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self.mcp_base_url}/tools/list")
                response.raise_for_status()
                tools = response.json()
                logger.info(f"Discovered {len(tools)} tools from MCP server")
                return tools
            except Exception as e:
                logger.error(f"Failed to fetch MCP tools: {e}")
                return []

    def _create_pydantic_model_from_schema(self, tool_name: str, parameters: Dict[str, Any]) -> Type[BaseModel]:
        """Create a Pydantic model from JSON schema parameters"""
        properties = parameters.get('properties', {})
        required = parameters.get('required', [])
        
        field_definitions = {}
        for field_name, field_info in properties.items():
            field_type = str  # default
            if field_info.get('type') == 'integer':
                field_type = int
            elif field_info.get('type') == 'number':
                field_type = float
            elif field_info.get('type') == 'boolean':
                field_type = bool
            elif field_info.get('type') == 'array':
                field_type = List[str]
            
            # Make field optional if not required
            if field_name not in required:
                field_type = Optional[field_type]
            
            # Handle default values
            default_value = field_info.get('default', ...)
            field_definitions[field_name] = (field_type, default_value)
        
        return create_model(f"{tool_name}Args", **field_definitions)

    def _create_currency_graph(self) -> StateGraph:
        """Create LangGraph workflow with dynamically discovered tools"""

        def should_continue(state: CurrencyState) -> str:
            messages = state["messages"]
            if not messages:
                return END
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return END

        async def call_model(state: CurrencyState):
            messages = state["messages"]
            
            # Build system message with discovered tools
            tool_descriptions = []
            for tool in self.available_tools:
                name = tool['name']
                desc = tool.get('description', 'No description')
                params = tool.get('parameters', {}).get('properties', {})
                param_str = ", ".join([f"{k}: {v.get('type', 'any')}" for k, v in params.items()])
                tool_descriptions.append(f"- {name}: {desc} (Parameters: {param_str})")
            
            system_message = SystemMessage(content=f"""
            You are a helpful currency conversion agent. Your job is to provide accurate exchange rates and currency conversions using real-time data.

            Available tools:
            {chr(10).join(tool_descriptions)}

            When users ask about currencies:
            1. Extract the currencies and amounts from their query
            2. Use appropriate tools to get real-time exchange rate data
            3. Present the information in a clear, user-friendly format

            Common currency codes: USD, EUR, GBP, JPY, AUD, CAD, CHF, CNY, INR, etc.
            Always use the tools to get real-time data. Present results in a friendly, conversational manner.
            """)
            
            full_messages = [system_message] + messages

            # Create tool objects from discovered tools
            tools = []
            for tool_info in self.available_tools:
                tool_name = tool_info['name']
                tool_desc = tool_info.get('description', '')
                tool_schema = self.tool_schemas.get(tool_name)
                
                def make_tool(name, description, args_schema):
                    async def tool_wrapper(**kwargs):
                        return f"Tool '{name}' would be called with: {kwargs}"
                    return Tool.from_function(
                        func=tool_wrapper,
                        name=name,
                        description=description,
                        args_schema=args_schema
                    )
                
                tools.append(make_tool(tool_name, tool_desc, tool_schema))

            llm_with_tools = self.llm.bind_tools(tools)
            response = llm_with_tools.invoke(full_messages)
            return {"messages": [response]}

        async def call_tools(state: CurrencyState):
            messages = state["messages"]
            if not messages:
                return {"messages": []}
            last_message = messages[-1]
            tool_messages = []

            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call.get("id", "")
                    
                    logger.info(f"Calling MCP tool: {tool_name} with args: {tool_args}")
                    result = await self._call_mcp_tool(tool_name, tool_args)
                    
                    # Format the result based on tool type
                    if "error" in result:
                        content = f"Error: {result['error']}"
                    else:
                        # Use tool-specific formatting if available
                        if tool_name == "convertCurrency":
                            content = f"Conversion result: {result.get('result', 'N/A')} {result.get('to_currency', '')} (Rate: {result.get('rate', 'N/A')})"
                        elif tool_name == "getExchangeRate":
                            content = f"Exchange rate: 1 {result.get('from_currency', '')} = {result.get('rate', 'N/A')} {result.get('to_currency', '')}"
                        elif tool_name == "getMultipleRates":
                            rates = result.get('rates', {})
                            rate_strings = [f"{curr}: {rate}" for curr, rate in rates.items()]
                            content = f"Rates from {result.get('base_currency', '')}: {', '.join(rate_strings)}"
                        else:
                            # Generic formatting for unknown tools
                            content = json.dumps(result, indent=2)
                    
                    tool_message = ToolMessage(content=content, tool_call_id=tool_call_id)
                    tool_messages.append(tool_message)
            return {"messages": tool_messages}

        workflow = StateGraph(CurrencyState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    async def initialize(self):
        """Initialize by discovering available tools from MCP server"""
        if self._initialized:
            return
        
        logger.info("Initializing currency agent with dynamic tool discovery...")
        
        # Discover available tools
        self.available_tools = await self._get_available_tools()
        
        # Create Pydantic models for each tool
        for tool in self.available_tools:
            tool_name = tool['name']
            parameters = tool.get('parameters', {})
            self.tool_schemas[tool_name] = self._create_pydantic_model_from_schema(tool_name, parameters)
        
        # Create the graph with discovered tools
        self.graph = self._create_currency_graph()
        self._initialized = True
        
        logger.info(f"Currency agent initialized with tools: {[t['name'] for t in self.available_tools]}")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        if not self._initialized:
            await self.initialize()
            
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()

        try:
            user_message = ""
            for part in context.message.parts:
                if isinstance(part.root, TextPart):
                    user_message += part.root.text
            if not user_message.strip():
                raise ValueError("No text content found in message")
                
            logger.info(f"Processing currency request: {user_message}")
            updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="Processing your currency request..."))
                ]),
            )
            
            initial_state = CurrencyState(messages=[HumanMessage(content=user_message)])
            final_state = await self.graph.ainvoke(initial_state)
            
            if final_state["messages"]:
                final_message = final_state["messages"][-1]
                response_text = final_message.content
            else:
                response_text = "I apologize, but I couldn't process your currency request."
                
            logger.info(f"Currency agent response: {response_text}")
            response_parts = [Part(root=TextPart(text=response_text))]
            updater.add_artifact(response_parts)
            updater.complete()

        except Exception as e:
            logger.error(f"Error in currency agent execution: {e}", exc_info=True)
            error_message = f"I apologize, but I encountered an error while processing your currency request: {str(e)}"
            error_parts = [Part(root=TextPart(text=error_message))]
            updater.add_artifact(error_parts)
            updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())