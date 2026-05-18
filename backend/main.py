# FastAPI = the web framework
# Request = lets us access raw request data if needed
from fastapi import FastAPI

# CORSMiddleware = allows the React frontend (on a different port)
# to talk to this backend
# Without CORS, browser blocks all requests from frontend → backend
from fastapi.middleware.cors import CORSMiddleware

# our two routers
from routers.documents import router as documents_router
from routers.chat import router as chat_router

# our vector store (needs to connect on startup)
from services.vector_store import vector_store

# settings
from core.config import get_settings

settings = get_settings()


# =============================================================================
# CREATE THE APP
# =============================================================================

app = FastAPI(
    title=settings.APP_TITLE,

    # description shows up in the auto-generated API docs at /docs
    description="""
    Multi-modal RAG Chatbot API

    Endpoints:
    - POST /documents/upload  → upload PDF, image, CSV, Excel
    - POST /documents/search  → semantic search across documents
    - POST /chat              → ask a question, get a full answer
    - WS   /chat/ws           → ask a question, get streaming answer
    """,

    version="1.0.0",

    # /docs = interactive Swagger UI (test all endpoints in browser)
    # /redoc = alternative cleaner docs
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# CORS — allow frontend to talk to backend
# =============================================================================
# CORS = Cross-Origin Resource Sharing
#
# Browser security rule: a webpage at localhost:3000 (React)
# cannot call an API at localhost:8000 (FastAPI) unless the API
# explicitly says "I allow requests from localhost:3000"
#
# In production, replace "*" with your actual frontend domain
# e.g. "https://myapp.com"

app.add_middleware(
    CORSMiddleware,

    # allow_origins = which domains can call this API
    allow_origins=["*"],        # * = allow ALL origins (fine for development)

    # allow frontend to send cookies and auth headers
    allow_credentials=True,

    # allow all HTTP methods (GET, POST, DELETE etc.)
    allow_methods=["*"],

    # allow all headers (Content-Type, Authorization etc.)
    allow_headers=["*"],
)


# =============================================================================
# STARTUP — runs once when the app boots
# =============================================================================

@app.on_event("startup")
async def startup():
    """
    Called automatically when FastAPI starts.
    We connect to Pinecone here so it's ready before any request comes in.

    Order matters:
      1. App starts
      2. startup() runs → connects to Pinecone
      3. App is ready to accept requests
    """
    print("Starting up...")
    await vector_store.connect()
    print("Connected to Pinecone. App is ready.")


# =============================================================================
# SHUTDOWN — runs once when the app stops
# =============================================================================

@app.on_event("shutdown")
async def shutdown():
    """
    Called automatically when FastAPI shuts down (Ctrl+C).
    Clean up any connections here.
    """
    print("Shutting down...")


# =============================================================================
# ATTACH ROUTERS
# =============================================================================
# Routers are mini apps — we attach them here so their routes become active
#
# documents_router has prefix="/documents" → routes become /documents/upload etc.
# chat_router     has prefix="/chat"       → routes become /chat and /chat/ws

app.include_router(documents_router)
app.include_router(chat_router)


# =============================================================================
# ROOT ENDPOINT — health check
# =============================================================================

@app.get("/")
async def root():
    """
    Simple health check endpoint.
    Visit http://localhost:8000 to confirm the app is running.
    """
    return {
        "status": "running",
        "app":    settings.APP_TITLE,
        "docs":   "http://localhost:8000/docs",
    }


# =============================================================================
# RUN THE APP  (only when running this file directly)
# =============================================================================

# This block only runs when you do: python main.py
# When using uvicorn command it is ignored
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",   # accept connections from any IP
        port=8000,
        reload=True,       # auto-restart when you save a file (dev only)
    )