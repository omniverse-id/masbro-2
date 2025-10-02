import os
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Dict, Any, Optional

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REFERER_URL = os.getenv("YOUR_SITE_URL", "http://localhost:8000")
X_TITLE = os.getenv("YOUR_SITE_NAME", "FastAPI OpenRouter App")

if not OPENROUTER_API_KEY:
    raise RuntimeError("FATAL: OPENROUTER_API_KEY tidak ditemukan di environment variables.")

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    http_client=httpx.Client(timeout=None)
)

app = FastAPI(title="OpenRouter Chatbot Backend")

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: str | List[ContentPart]

class ChatRequest(BaseModel):
    model: str = "openai/gpt-4o"
    messages: List[Message]

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):

    def format_messages_for_openai(messages: List[Message]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for msg in messages:
            content_value = msg.content
            if isinstance(content_value, list):
                content_list = []
                for part in content_value:
                    if part.type == "text":
                        content_list.append({"type": "text", "text": part.text})
                    elif part.type == "image_url" and part.image_url and part.image_url.get("url"):
                        content_list.append({"type": "image_url", "image_url": part.image_url})
                formatted_messages.append({"role": msg.role, "content": content_list})
            elif isinstance(content_value, str):
                formatted_messages.append({"role": msg.role, "content": content_value})
        return formatted_messages

    def stream_generator(messages: List[Dict[str, Any]]):
        try:
            stream = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": REFERER_URL,
                    "X-Title": X_TITLE,
                },
                model=request.model,
                messages=messages,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"Error during streaming: {e}")
            yield f"\n\n[ERROR: {str(e)}]"

    formatted_messages = format_messages_for_openai(request.messages)

    return StreamingResponse(
        stream_generator(messages=formatted_messages),
        media_type="text/plain"
    )