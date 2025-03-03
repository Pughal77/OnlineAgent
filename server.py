from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="OpenAI API Proxy",
    description="A FastAPI application to interact with OpenAI's API endpoints",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Base URL for OpenAI API
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify the API key."""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is missing")
    # You can implement your own API key validation logic here
    # For now, we'll just check if it's non-empty
    return api_key

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the OpenAI API Proxy. See /docs for API documentation."}

@app.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Proxy for OpenAI's chat completions endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                json=request.dict(exclude_none=True),
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/completions")
async def completions(
    request: CompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Proxy for OpenAI's completions endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OPENAI_BASE_URL}/completions",
                json=request.dict(exclude_none=True),
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models(
    api_key: str = Depends(verify_api_key)
):
    """Proxy for OpenAI's list models endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{OPENAI_BASE_URL}/models",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}")
async def get_model(
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Proxy for OpenAI's get model endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{OPENAI_BASE_URL}/models/{model_id}",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)