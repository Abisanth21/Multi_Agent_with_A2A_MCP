
# currency_agent/mcp_server.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import os
import logging
import requests
import uvicorn
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")

app = FastAPI(title="Currency MCP Server", version="1.0.0")

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
        "status": "Currency MCP Server is running",
        "note": "Agent is mounted at /agent",
        "redirect": "/agent/"
    })

@app.post("/")
async def redirect_to_agent():
    return RedirectResponse(url="/agent", status_code=307)

@app.get("/agent/")
async def agent_root():
    return JSONResponse({
        "status": "Currency MCP Server is running", 
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
        "exchange_api_configured": bool(EXCHANGE_API_KEY),
        "available_tools": ["convertCurrency", "getExchangeRate", "getMultipleRates"]
    })

@app.get("/tools/list")
async def list_tools():
    tools = [
        {
            "name": "convertCurrency",
            "description": "Convert an amount from one currency to another using real-time exchange rates",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert"},
                    "from_currency": {"type": "string", "description": "Source currency code (e.g., USD)"},
                    "to_currency": {"type": "string", "description": "Target currency code (e.g., EUR)"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        },
        {
            "name": "getExchangeRate",
            "description": "Get current exchange rate between two currencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string", "description": "Source currency code"},
                    "to_currency": {"type": "string", "description": "Target currency code"}
                },
                "required": ["from_currency", "to_currency"]
            }
        },
        {
            "name": "getMultipleRates",
            "description": "Get exchange rates from one base currency to multiple target currencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_currency": {"type": "string", "description": "Base currency code"},
                    "target_currencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target currency codes",
                        "default": ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF"]
                    }
                },
                "required": ["base_currency"]
            }
        }
    ]
    return JSONResponse(tools)

def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    """Convert currency with improved error handling"""
    if not EXCHANGE_API_KEY:
        return {
            "error": "EXCHANGE_API_KEY not configured",
            "message": "Please set the EXCHANGE_API_KEY environment variable"
        }

    try:
        amount = float(amount)
        if amount <= 0:
            return {"error": "Amount must be greater than 0"}
    except (ValueError, TypeError):
        return {"error": "Invalid amount provided"}

    if not from_currency or not to_currency:
        return {"error": "Both from_currency and to_currency are required"}

    from_currency = str(from_currency).upper().strip()
    to_currency = str(to_currency).upper().strip()

    if from_currency == to_currency:
        return {
            "result": amount,
            "rate": 1.0,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "note": "Source and target currencies are the same."
        }

    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/pair/{from_currency}/{to_currency}/{amount}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") == "success":
            return {
                "result": data.get("conversion_result"),
                "rate": data.get("conversion_rate"),
                "from_currency": from_currency,
                "to_currency": to_currency
            }
        else:
            return {"error": data.get("error-type", "Unknown error")}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def get_exchange_rate(from_currency: str, to_currency: str) -> Dict[str, Any]:
    """Get exchange rate with improved error handling"""
    if not EXCHANGE_API_KEY:
        return {"error": "EXCHANGE_API_KEY not configured"}

    if not from_currency or not to_currency:
        return {"error": "Both from_currency and to_currency are required"}

    from_currency = str(from_currency).upper().strip()
    to_currency = str(to_currency).upper().strip()

    if from_currency == to_currency:
        return {"rate": 1.0, "from_currency": from_currency, "to_currency": to_currency}

    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/pair/{from_currency}/{to_currency}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") == "success":
            return {
                "rate": data.get("conversion_rate"),
                "from_currency": from_currency,
                "to_currency": to_currency
            }
        else:
            return {"error": data.get("error-type", "Unknown error")}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def get_multiple_rates(base_currency: str, target_currencies: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get multiple rates with improved error handling"""
    if not EXCHANGE_API_KEY:
        return {"error": "EXCHANGE_API_KEY not configured"}

    if not base_currency:
        return {"error": "base_currency is required"}

    base_currency = str(base_currency).upper().strip()
    if target_currencies is None:
        target_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF"]
    else:
        target_currencies = [str(c).upper().strip() for c in target_currencies]

    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/{base_currency}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") == "success":
            rates = data.get("conversion_rates", {})
            filtered = {c: rates.get(c) for c in target_currencies if c in rates}
            return {
                "base_currency": base_currency,
                "rates": filtered
            }
        else:
            return {"error": data.get("error-type", "Unknown error")}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
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
        
        if tool_name == "convertCurrency":
            result = convert_currency(
                params.get("amount"),
                params.get("from_currency"),
                params.get("to_currency")
            )
        elif tool_name == "getExchangeRate":
            result = get_exchange_rate(
                params.get("from_currency"),
                params.get("to_currency")
            )
        elif tool_name == "getMultipleRates":
            result = get_multiple_rates(
                params.get("base_currency"),
                params.get("target_currencies")
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
    uvicorn.run(app, host="localhost", port=8001)