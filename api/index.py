from http.server import BaseHTTPRequestHandler
from intent_router_api import app
import json

def handler(event, context):
    """Vercel serverless function handler"""
    try:
        # Convert Vercel event to ASGI scope
        scope = {
            "type": "http",
            "method": event.get("httpMethod", "GET"),
            "path": event.get("path", "/"),
            "headers": event.get("headers", {}),
            "query_string": event.get("queryStringParameters", {}),
            "body": event.get("body", ""),
        }
        
        # Create ASGI application
        async def asgi_app(scope, receive, send):
            await app(scope, receive, send)
        
        # Handle the request
        response = asgi_app(scope, None, None)
        
        return {
            "statusCode": 200,
            "body": json.dumps(response),
            "headers": {
                "Content-Type": "application/json"
            }
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {
                "Content-Type": "application/json"
            }
        } 