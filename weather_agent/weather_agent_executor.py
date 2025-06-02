# # weather_agent/weather_agent_executor.py
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

# class GetWeatherArgs(BaseModel):
#     city: str
#     unit: Optional[str] = "metric"

# class GetWeatherForecastArgs(BaseModel):
#     city: str
#     days: Optional[int] = 5
#     unit: Optional[str] = "metric"

# logger = logging.getLogger(__name__)

# class WeatherState(TypedDict):
#     messages: Annotated[list, add_messages]

# class WeatherAgentExecutor(AgentExecutor):
#     """Weather Agent using LangGraph with MCP integration."""

#     def __init__(self, mcp_base_url: str = "http://localhost:3000"):
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
#                 if tool_name == "getWeather":
#                     cleaned_args = {
#                         "city": str(arguments.get("city", "")).strip(),
#                         "unit": str(arguments.get("unit", "metric")).lower()
#                     }
#                 elif tool_name == "getWeatherForecast":
#                     cleaned_args = {
#                         "city": str(arguments.get("city", "")).strip(),
#                         "days": int(arguments.get("days", 5)),
#                         "unit": str(arguments.get("unit", "metric")).lower()
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

#     def _create_weather_graph(self) -> StateGraph:

#         def should_continue(state: WeatherState) -> str:
#             messages = state["messages"]
#             if not messages:
#                 return END
#             last_message = messages[-1]
#             if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#                 return "tools"
#             return END

#         async def call_model(state: WeatherState):
#             messages = state["messages"]
#             system_message = SystemMessage(content="""
#             You are a helpful weather agent. Your job is to provide accurate weather information and forecasts using real-time data.

#             When users ask about weather:
#             1. Extract the location from their query
#             2. Use appropriate tools to get weather data
#             3. Present the information in a clear, user-friendly format
#             4. Include relevant details like temperature, conditions, humidity, wind, etc.

#             Available tools:
#             - getWeather: Get current weather information for a specific location
#             - getWeatherForecast: Get weather forecast for a specific location (up to 10 days)

#             If location is not clear, ask for clarification.
#             Always use the tools to get real-time data. Present results in a friendly, conversational manner.
            
#             Example usage:
#             - For "What's the weather in London?": use getWeather with city="London"
#             - For "5-day forecast for New York": use getWeatherForecast with city="New York", days=5
#             - For temperature in Celsius: use unit="metric"
#             - For temperature in Fahrenheit: use unit="imperial"
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
#                 make_tool("getWeather", 
#                          "Get current weather information for a specific location", 
#                          GetWeatherArgs),
#                 make_tool("getWeatherForecast", 
#                          "Get weather forecast for a specific location", 
#                          GetWeatherForecastArgs)
#             ]

#             llm_with_tools = self.llm.bind_tools(tools)
#             response = llm_with_tools.invoke(full_messages)
#             return {"messages": [response]}

#         async def call_tools(state: WeatherState):
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
#                         if tool_name == "getWeather":
#                             content = self._format_current_weather(result)
#                         elif tool_name == "getWeatherForecast":
#                             content = self._format_weather_forecast(result)
#                         else:
#                             content = json.dumps(result)
                    
#                     tool_message = ToolMessage(content=content, tool_call_id=tool_call_id)
#                     tool_messages.append(tool_message)
#             return {"messages": tool_messages}

#         workflow = StateGraph(WeatherState)
#         workflow.add_node("agent", call_model)
#         workflow.add_node("tools", call_tools)
#         workflow.set_entry_point("agent")
#         workflow.add_conditional_edges("agent", should_continue)
#         workflow.add_edge("tools", "agent")
#         return workflow.compile()

#     def _format_current_weather(self, data: Dict[str, Any]) -> str:
#         """Format current weather data for better readability"""
#         try:
#             location = f"{data.get('location', '')}, {data.get('region', '')}, {data.get('country', '')}"
#             temp_c = data.get('temperature_c', 'N/A')
#             temp_f = data.get('temperature_f', 'N/A')
#             condition = data.get('condition', 'N/A')
#             humidity = data.get('humidity', 'N/A')
#             wind_kph = data.get('wind_kph', 'N/A')
#             feels_like_c = data.get('feels_like_c', 'N/A')
#             uv_index = data.get('uv_index', 'N/A')
            
#             return f"Current weather in {location}: {temp_c}°C ({temp_f}°F), {condition}. " \
#                    f"Feels like {feels_like_c}°C. Humidity: {humidity}%, Wind: {wind_kph} km/h, UV Index: {uv_index}"
#         except Exception:
#             return json.dumps(data)

#     def _format_weather_forecast(self, data: Dict[str, Any]) -> str:
#         """Format weather forecast data for better readability"""
#         try:
#             location = f"{data.get('location', '')}, {data.get('region', '')}, {data.get('country', '')}"
#             forecasts = data.get('forecasts', [])
            
#             if not forecasts:
#                 return f"No forecast data available for {location}"
            
#             result = f"Weather forecast for {location}:\n"
#             for forecast in forecasts:
#                 date = forecast.get('date', '')
#                 max_temp = forecast.get('max_temp_c', 'N/A')
#                 min_temp = forecast.get('min_temp_c', 'N/A')
#                 condition = forecast.get('condition', 'N/A')
#                 rain_chance = forecast.get('chance_of_rain', 'N/A')
                
#                 result += f"• {date}: {condition}, High: {max_temp}°C, Low: {min_temp}°C, Rain: {rain_chance}%\n"
            
#             return result.strip()
#         except Exception:
#             return json.dumps(data)

#     async def initialize(self):
#         if self._initialized:
#             return
#         try:
#             tools = await self._get_available_tools()
#             logger.info(f"Found {len(tools)} MCP tools available")
#         except Exception as e:
#             logger.warning(f"Could not connect to MCP server: {e}")
#         self.graph = self._create_weather_graph()
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
#             logger.info(f"Processing weather request: {user_message}")
#             updater.update_status(
#                 TaskState.working,
#                 message=updater.new_agent_message([
#                     Part(root=TextPart(text="Processing your weather request..."))
#                 ]),
#             )
#             initial_state = WeatherState(messages=[HumanMessage(content=user_message)])
#             final_state = await self.graph.ainvoke(initial_state)
#             if final_state["messages"]:
#                 final_message = final_state["messages"][-1]
#                 response_text = final_message.content
#             else:
#                 response_text = "I apologize, but I couldn't process your weather request."
#             logger.info(f"Weather agent response: {response_text}")
#             response_parts = [Part(root=TextPart(text=response_text))]
#             updater.add_artifact(response_parts)
#             updater.complete()

#         except Exception as e:
#             logger.error(f"Error in weather agent execution: {e}", exc_info=True)
#             error_message = f"I apologize, but I encountered an error while processing your weather request: {str(e)}"
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

class WeatherState(TypedDict):
    messages: Annotated[list, add_messages]

class WeatherAgentExecutor(AgentExecutor):
    """Weather Agent using LangGraph with dynamic MCP tool discovery."""

    def __init__(self, mcp_base_url: str = "http://localhost:3000"):
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
            
            field_definitions[field_name] = (field_type, field_info.get('default', ...))
        
        return create_model(f"{tool_name}Args", **field_definitions)

    def _create_weather_graph(self) -> StateGraph:
        """Create LangGraph workflow with dynamically discovered tools"""

        def should_continue(state: WeatherState) -> str:
            messages = state["messages"]
            if not messages:
                return END
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return END

        async def call_model(state: WeatherState):
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
            You are a helpful weather agent. Your job is to provide accurate weather information using real-time data.

            Available tools:
            {chr(10).join(tool_descriptions)}

            When users ask about weather:
            1. Extract the location from their query
            2. Use appropriate tools to get weather data
            3. Present the information in a clear, user-friendly format

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

        async def call_tools(state: WeatherState):
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
                        if tool_name == "getWeather":
                            content = self._format_current_weather(result)
                        elif tool_name == "getWeatherForecast":
                            content = self._format_weather_forecast(result)
                        else:
                            # Generic formatting for unknown tools
                            content = json.dumps(result, indent=2)
                    
                    tool_message = ToolMessage(content=content, tool_call_id=tool_call_id)
                    tool_messages.append(tool_message)
            return {"messages": tool_messages}

        workflow = StateGraph(WeatherState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def _format_current_weather(self, data: Dict[str, Any]) -> str:
        """Format current weather data for better readability"""
        try:
            location = f"{data.get('location', '')}, {data.get('region', '')}, {data.get('country', '')}"
            temp_c = data.get('temperature_c', 'N/A')
            temp_f = data.get('temperature_f', 'N/A')
            condition = data.get('condition', 'N/A')
            humidity = data.get('humidity', 'N/A')
            wind_kph = data.get('wind_kph', 'N/A')
            feels_like_c = data.get('feels_like_c', 'N/A')
            uv_index = data.get('uv_index', 'N/A')
            
            return f"Current weather in {location}: {temp_c}°C ({temp_f}°F), {condition}. " \
                   f"Feels like {feels_like_c}°C. Humidity: {humidity}%, Wind: {wind_kph} km/h, UV Index: {uv_index}"
        except Exception:
            return json.dumps(data)

    def _format_weather_forecast(self, data: Dict[str, Any]) -> str:
        """Format weather forecast data for better readability"""
        try:
            location = f"{data.get('location', '')}, {data.get('region', '')}, {data.get('country', '')}"
            forecasts = data.get('forecasts', [])
            
            if not forecasts:
                return f"No forecast data available for {location}"
            
            result = f"Weather forecast for {location}:\n"
            for forecast in forecasts:
                date = forecast.get('date', '')
                max_temp = forecast.get('max_temp_c', 'N/A')
                min_temp = forecast.get('min_temp_c', 'N/A')
                condition = forecast.get('condition', 'N/A')
                rain_chance = forecast.get('chance_of_rain', 'N/A')
                
                result += f"• {date}: {condition}, High: {max_temp}°C, Low: {min_temp}°C, Rain: {rain_chance}%\n"
            
            return result.strip()
        except Exception:
            return json.dumps(data)

    async def initialize(self):
        """Initialize by discovering available tools from MCP server"""
        if self._initialized:
            return
        
        logger.info("Initializing weather agent with dynamic tool discovery...")
        
        # Discover available tools
        self.available_tools = await self._get_available_tools()
        
        # Create Pydantic models for each tool
        for tool in self.available_tools:
            tool_name = tool['name']
            parameters = tool.get('parameters', {})
            self.tool_schemas[tool_name] = self._create_pydantic_model_from_schema(tool_name, parameters)
        
        # Create the graph with discovered tools
        self.graph = self._create_weather_graph()
        self._initialized = True
        
        logger.info(f"Weather agent initialized with tools: {[t['name'] for t in self.available_tools]}")

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
                
            logger.info(f"Processing weather request: {user_message}")
            updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="Processing your weather request..."))
                ]),
            )
            
            initial_state = WeatherState(messages=[HumanMessage(content=user_message)])
            final_state = await self.graph.ainvoke(initial_state)
            
            if final_state["messages"]:
                final_message = final_state["messages"][-1]
                response_text = final_message.content
            else:
                response_text = "I apologize, but I couldn't process your weather request."
                
            logger.info(f"Weather agent response: {response_text}")
            response_parts = [Part(root=TextPart(text=response_text))]
            updater.add_artifact(response_parts)
            updater.complete()

        except Exception as e:
            logger.error(f"Error in weather agent execution: {e}", exc_info=True)
            error_message = f"I apologize, but I encountered an error while processing your weather request: {str(e)}"
            error_parts = [Part(root=TextPart(text=error_message))]
            updater.add_artifact(error_parts)
            updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())