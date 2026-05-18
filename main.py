from ollama import chat
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

# 1. Define a Pydantic model to handle the incoming JSON payload
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/chat")
async def chatbot_endpoint(request: ChatRequest):
    
    # 2. Wrap your streaming logic inside a generator function
    def generate_chat():
        stream = chat(
            model='gemma4:e2b',
            messages=request.messages,
            think=True,
            stream=True,
        )

        in_thinking = False

        for chunk in stream:
            # Yield instead of print to stream the chunks over HTTP
            if chunk.message.thinking and not in_thinking:
                in_thinking = True
                yield 'Thinking:\n'

            if chunk.message.thinking:
                yield chunk.message.thinking
            elif chunk.message.content:
                if in_thinking:
                    yield '\n\nAnswer:\n'
                    in_thinking = False
                yield chunk.message.content

    # 3. Return the generator wrapped in a StreamingResponse
    return StreamingResponse(generate_chat(), media_type="text/plain")