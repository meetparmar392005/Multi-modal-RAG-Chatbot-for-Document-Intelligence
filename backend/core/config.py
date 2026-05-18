# pydantic-settings lets us read values from the .env file automatically
from pydantic_settings import BaseSettings

# lru_cache means this function runs only ONCE and reuses the result
# (we don't want to reload .env on every request — wasteful)
from functools import lru_cache


class Settings(BaseSettings):
    """
    Every variable here maps directly to a line in your .env file.
    If the .env line is missing, the default value is used instead.
    """

    # Human-readable name used by FastAPI and the health check endpoint
    APP_TITLE: str = "Multi-modal RAG Chatbot API"

    # ── LLM settings ─────────────────────────────────────────────────────────
    # Your OpenAI secret key — never hardcode this, always from .env
    OPENAI_API_KEY: str = ""

    # Which OpenAI model answers the user's questions
    # gpt-4o understands both text AND images (multi-modal)
    LLM_MODEL: str = "gpt-4o"

    # Which model converts text into a vector (list of numbers)
    # text-embedding-3-small is fast and cheap, good for starting out
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ── Vector Database settings ──────────────────────────────────────────────
    # Your Pinecone secret key
    PINECONE_API_KEY: str = ""

    # The name of the "table" inside Pinecone where embeddings are stored
    PINECONE_INDEX_NAME: str = "rag-index"

    # ── App behaviour settings ────────────────────────────────────────────────
    # When we split a document into pieces, each piece is max 800 characters
    # Smaller = more precise retrieval, Larger = more context per chunk
    CHUNK_SIZE: int = 800

    # How many characters overlap between two consecutive chunks
    # Overlap prevents losing meaning at chunk boundaries
    # Example: chunk1 ends at char 800, chunk2 starts at char 700
    CHUNK_OVERLAP: int = 100

    # When user asks a question, how many chunks to pull back from Pinecone
    TOP_K_RESULTS: int = 5

    class Config:
        # Tell pydantic WHERE to find the .env file
        env_file = ".env"


# This function returns the settings object
# @lru_cache ensures it's created only once for the whole app lifetime
@lru_cache
def get_settings() -> Settings:
    return Settings()