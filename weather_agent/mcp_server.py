# weather_agent/mcp_server.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import os
import logging
import requests
import uvicorn
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

app = FastAPI(title="Weather MCP Server", version="1.0.0")

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return JSONResponse({
        "status": "Weather MCP Server is running",
        "note": "Agent is mounted at /agent",
        "redirect": "/agent/"
    })

@app.post("/")
async def redirect_to_agent():
    return RedirectResponse(url="/agent", status_code=307)

@app.get("/agent/")
async def agent_root():
    return JSONResponse({
        "status": "Weather MCP Server is running", 
        "version": "1.0.0",
        "endpoints": {
            "tools_list": "/tools/list",
            "tools_call": "/tools/call",
            "health": "/health"
        }
    })

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "weather_api_configured": bool(WEATHER_API_KEY),
        "available_tools": ["getWeather", "getWeatherForecast"]
    })

@app.get("/tools/list")
async def list_tools():
    tools = [
        {
            "name": "getWeather",
            "description": "Get current weather information for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name or location (e.g., 'London', 'New York, NY')"},
                    "unit": {"type": "string", "description": "Temperature unit", "enum": ["metric", "imperial"], "default": "metric"}
                },
                "required": ["city"]
            }
        },
        {
            "name": "getWeatherForecast",
            "description": "Get weather forecast for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name or location"},
                    "days": {"type": "integer", "description": "Number of forecast days (1-10)", "default": 5, "minimum": 1, "maximum": 10},
                    "unit": {"type": "string", "description": "Temperature unit", "enum": ["metric", "imperial"], "default": "metric"}
                },
                "required": ["city"]
            }
        }
    ]
    return JSONResponse(tools)

def get_weather(city: str, unit: str = "metric") -> Dict[str, Any]:
    """Get current weather information with improved error handling"""
    if not WEATHER_API_KEY:
        return {
            "error": "WEATHER_API_KEY not configured",
            "message": "Please set the WEATHER_API_KEY environment variable"
        }

    if not city or not city.strip():
        return {"error": "City name is required"}

    city = str(city).strip()

    try:
        base_url = "http://api.weatherapi.com/v1/current.json"
        params = {
            'key': WEATHER_API_KEY,
            'q': city,
            'aqi': 'no'
        }
        
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract relevant weather information
        weather_info = {
            'location': data['location']['name'],
            'region': data['location']['region'],
            'country': data['location']['country'],
            'temperature_c': data['current']['temp_c'],
            'temperature_f': data['current']['temp_f'],
            'condition': data['current']['condition']['text'],
            'humidity': data['current']['humidity'],
            'wind_kph': data['current']['wind_kph'],
            'wind_mph': data['current']['wind_mph'],
            'feels_like_c': data['current']['feelslike_c'],
            'feels_like_f': data['current']['feelslike_f'],
            'visibility_km': data['current']['vis_km'],
            'uv_index': data['current']['uv'],
            'last_updated': data['current']['last_updated']
        }
        
        return weather_info
        
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - weather service unavailable"}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return {"error": f"Invalid location: {city}"}
        elif e.response.status_code == 401:
            return {"error": "Invalid API key"}
        elif e.response.status_code == 403:
            return {"error": "API key exceeded quota"}
        else:
            return {"error": f"Weather service error: {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except KeyError as e:
        return {"error": f"Unexpected response format: missing {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def get_weather_forecast(city: str, days: int = 5, unit: str = "metric") -> Dict[str, Any]:
    """Get weather forecast with improved error handling"""
    if not WEATHER_API_KEY:
        return {
            "error": "WEATHER_API_KEY not configured",
            "message": "Please set the WEATHER_API_KEY environment variable"
        }

    if not city or not city.strip():
        return {"error": "City name is required"}

    city = str(city).strip()
    
    try:
        days = int(days)
        if days < 1 or days > 10:
            return {"error": "Days must be between 1 and 10"}
    except (ValueError, TypeError):
        return {"error": "Invalid days parameter"}

    try:
        base_url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            'key': WEATHER_API_KEY,
            'q': city,
            'days': days,
            'aqi': 'no',
            'alerts': 'no'
        }
        
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract forecast information
        forecasts = []
        for day in data['forecast']['forecastday']:
            forecast = {
                'date': day['date'],
                'max_temp_c': day['day']['maxtemp_c'],
                'max_temp_f': day['day']['maxtemp_f'],
                'min_temp_c': day['day']['mintemp_c'],
                'min_temp_f': day['day']['mintemp_f'],
                'avg_temp_c': day['day']['avgtemp_c'],
                'avg_temp_f': day['day']['avgtemp_f'],
                'condition': day['day']['condition']['text'],
                'max_wind_kph': day['day']['maxwind_kph'],
                'max_wind_mph': day['day']['maxwind_mph'],
                'avg_humidity': day['day']['avghumidity'],
                'chance_of_rain': day['day']['daily_chance_of_rain'],
                'chance_of_snow': day['day']['daily_chance_of_snow'],
                'uv_index': day['day']['uv']
            }
            forecasts.append(forecast)
        
        return {
            'location': data['location']['name'],
            'region': data['location']['region'],
            'country': data['location']['country'],
            'forecasts': forecasts
        }
        
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - weather service unavailable"}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return {"error": f"Invalid location: {city}"}
        elif e.response.status_code == 401:
            return {"error": "Invalid API key"}
        elif e.response.status_code == 403:
            return {"error": "API key exceeded quota"}
        else:
            return {"error": f"Weather service error: {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except KeyError as e:
        return {"error": f"Unexpected response format: missing {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@app.post("/tools/call")
async def call_tool(request: Request):
    try:
        body = await request.json()
        tool_name = body.get("tool") or body.get("name")
        # Handle both 'parameters' and 'arguments' keys for flexibility
        params = body.get("parameters") or body.get("arguments") or {}
        
        logger.info(f"Tool call received: {tool_name} with params: {params}")
        
        if tool_name == "getWeather":
            result = get_weather(
                params.get("city"),
                params.get("unit", "metric")
            )
        elif tool_name == "getWeatherForecast":
            result = get_weather_forecast(
                params.get("city"),
                params.get("days", 5),
                params.get("unit", "metric")
            )
        else:
            return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=400)
        
        logger.info(f"Tool result: {result}")
        
        if "error" in result:
            return JSONResponse({"error": result["error"]}, status_code=400)
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"call_tool error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse({"error": "Internal server error"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8002)