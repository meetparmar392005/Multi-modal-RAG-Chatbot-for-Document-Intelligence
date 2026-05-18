# APIRouter = creates a mini router we attach to main.py
# WebSocket = real-time two-way connection (for streaming responses)
# WebSocketDisconnect = raised when browser closes the connection
# HTTPException = returns proper HTTP errors
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

# our request/response schemas
from models.schemas import ChatRequest, ChatResponse

# services we need
from services.embeddings import encode, decode
from services.vector_store import vector_store

# settings
from core.config import get_settings

# json = serialize data to send over WebSocket
import json

settings = get_settings()

router = APIRouter(prefix="/chat", tags=["Chat"])


# =============================================================================
# WHAT IS RAG?
# =============================================================================
# RAG = Retrieval Augmented Generation
#
# Without RAG:
#   User: "What was our Q3 revenue?"
#   LLM:  "I don't know, I wasn't trained on your documents."
#
# With RAG:
#   1. Embed the question → search Pinecone → find relevant chunks
#   2. Build a prompt:
#      "Answer using ONLY this context: [chunks]
#       Question: What was our Q3 revenue?"
#   3. LLM reads the context and answers accurately
#
# The LLM doesn't need to "know" anything — it just reads and summarizes


# =============================================================================
# HELPER — Build the prompt
# =============================================================================

def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Combine retrieved chunks + user question into one prompt for the LLM.

    Why "ONLY the context below"?
    We want the LLM to answer from our documents only.
    This prevents hallucination — making up facts not in the documents.

    Example output:
    ------------------------------------
    You are a helpful assistant. Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't have enough information."

    CONTEXT:
    [Source: report.pdf | Page: 3]
    Q3 revenue was $2.4 million, up 12% from Q2...

    [Source: report.pdf | Page: 5]
    Revenue growth was driven by new enterprise clients...

    QUESTION:
    What was the Q3 revenue?
    ------------------------------------
    """
    # Format each chunk with its source info
    context_text = ""
    for chunk in context_chunks:
        context_text += (
            f"[Source: {chunk['source']} | Page: {chunk['page']}]\n"
            f"{chunk['text']}\n\n"
        )

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not found in the context, say "I don't have enough information to answer this."
Do not make up any information that is not in the context.

CONTEXT:
{context_text}
QUESTION:
{question}

ANSWER:"""

    return prompt


# =============================================================================
# POST /chat  — standard request/response (no streaming)
# =============================================================================

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question and get a full answer back at once.

    Flow:
      1. Embed the question
      2. Search Pinecone for relevant chunks
      3. Build a prompt with the chunks as context
      4. Send to GPT-4o and wait for full response
      5. Return answer + source chunks used

    Use this for simple integrations.
    Use WebSocket /chat/ws for streaming (better UX).

    Example request:
      { "question": "What were the key findings?", "top_k": 5 }

    Example response:
      {
        "answer": "The key findings were...",
        "sources": [
          {"text": "...", "score": 0.92, "source": "report.pdf", "page": 3}
        ]
      }
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # ── Step 1: Retrieve relevant chunks from Pinecone ────────────────────────
    try:
        chunks = await decode(
            query=request.question,
            vector_store=vector_store,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Please upload documents first."
        )

    # ── Step 2: Build prompt with retrieved context ───────────────────────────
    prompt = build_prompt(request.question, chunks)

    # ── Step 3: Send to LLM and get full response ─────────────────────────────
    try:
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful document assistant. "
                               "Only answer based on the provided context."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=1024,
            temperature=0.2,  # low temperature = focused, factual answers
        )
        answer = response.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {str(e)}")

    # ── Step 4: Return answer + sources ──────────────────────────────────────
    return ChatResponse(
        answer=answer,
        sources=chunks,
    )


# =============================================================================
# WEBSOCKET /chat/ws  — streaming response (token by token)
# =============================================================================

@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat responses.

    Why WebSocket instead of HTTP?
    HTTP: user waits 10 seconds → full answer appears at once (bad UX)
    WebSocket: answer streams word by word as LLM generates it (like ChatGPT)

    Message flow:
      Frontend sends:  { "question": "...", "top_k": 5 }
      Backend sends:   { "type": "token",  "content": "The " }
      Backend sends:   { "type": "token",  "content": "answer " }
      Backend sends:   { "type": "token",  "content": "is..." }
      Backend sends:   { "type": "sources","content": [...] }
      Backend sends:   { "type": "done",   "content": "" }
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Accept the WebSocket connection from the frontend
    await websocket.accept()

    try:
        # Keep connection open — handle multiple questions in one session
        while True:

            # ── Wait for a message from the frontend ──────────────────────────
            raw_message = await websocket.receive_text()

            try:
                data = json.loads(raw_message)
                question = data.get("question", "").strip()
                top_k = data.get("top_k", settings.TOP_K_RESULTS)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "Invalid message format. Send JSON with 'question' field."
                }))
                continue

            if not question:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "Question cannot be empty."
                }))
                continue

            # ── Step 1: Retrieve relevant chunks ─────────────────────────────
            try:
                chunks = await decode(
                    query=question,
                    vector_store=vector_store,
                    top_k=top_k,
                )
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"Retrieval failed: {str(e)}"
                }))
                continue

            # ── Step 2: Build prompt ──────────────────────────────────────────
            prompt = build_prompt(question, chunks)

            # ── Step 3: Stream LLM response token by token ───────────────────
            try:
                stream = await client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful document assistant. "
                                       "Only answer based on the provided context."
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_tokens=1024,
                    temperature=0.2,
                    stream=True,   # ← this enables token-by-token streaming
                )

                # Send each token to the frontend as it arrives
                async for chunk in stream:
                    token = chunk.choices[0].delta.content

                    if token:  # skip empty tokens
                        await websocket.send_text(json.dumps({
                            "type": "token",      # frontend appends this to the chat bubble
                            "content": token,
                        }))

            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"LLM streaming failed: {str(e)}"
                }))
                continue

            # ── Step 4: Send sources after answer is complete ─────────────────
            await websocket.send_text(json.dumps({
                "type": "sources",    # frontend shows these as citations
                "content": chunks,
            }))

            # ── Step 5: Signal that this response is fully done ───────────────
            await websocket.send_text(json.dumps({
                "type": "done",       # frontend stops showing typing indicator
                "content": "",
            }))

    except WebSocketDisconnect:
        # Browser closed the tab or lost connection — this is normal
        print("WebSocket client disconnected")