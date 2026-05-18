# Pydantic is a data validation library
# BaseModel = base class for all our schemas
# Field = lets us add extra rules to a field (min length, max length, description)
from pydantic import BaseModel, Field

# Optional means the field is not required — it has a default value
from typing import Optional


# =============================================================================
# UPLOAD SCHEMAS
# =============================================================================
# Used by: POST /upload endpoint
# When user uploads a file, this is what we send back as a response

class UploadResponse(BaseModel):
    """
    Returned after a document is successfully uploaded and processed.
    
    Example response:
    {
        "file_name": "report.pdf",
        "chunks_created": 12,
        "embeddings_stored": 12,
        "message": "File processed successfully"
    }
    """
    file_name: str            # original name of the uploaded file
    chunks_created: int       # how many text chunks we split the document into
    embeddings_stored: int    # how many vectors were saved to Pinecone
    message: str              # human readable success/error message


# =============================================================================
# EMBEDDING SCHEMAS
# =============================================================================
# Embeddings = converting text into a list of numbers (vector)
# This is how we make text mathematically searchable

class EmbedRequest(BaseModel):
    """
    Request body for manually embedding a piece of text.
    Useful for testing your embedding service directly.
    
    Example request:
    { "text": "What is machine learning?" }
    """
    # min_length=1 means empty strings are rejected automatically
    # max_length=10000 prevents someone sending a huge string
    text: str = Field(..., min_length=1, max_length=10_000)


class EmbedResponse(BaseModel):
    """
    Response after embedding a text.
    
    Example response:
    {
        "text": "What is machine learning?",
        "embedding": [0.012, -0.045, 0.089, ...],  # 1536 numbers
        "dimensions": 1536
    }
    """
    text: str                  # the original text you sent
    embedding: list[float]     # the vector — list of decimal numbers
    dimensions: int            # length of the vector (1536 for OpenAI small model)


# =============================================================================
# DECODE / SEARCH SCHEMAS
# =============================================================================
# Decode = given a vector, find the most similar text chunks in Pinecone
# This is called "semantic search" — search by meaning, not exact words

class SearchRequest(BaseModel):
    """
    Request body for semantic search.
    Send a plain text query → we embed it → search Pinecone → return matches.
    
    Example request:
    { "query": "what is the revenue for Q3?", "top_k": 5 }
    """
    query: str = Field(..., min_length=1)   # the user's search question
    top_k: Optional[int] = 5               # how many results to return


class SearchResponse(BaseModel):
    """
    Response from semantic search.
    
    Example response:
    {
        "query": "what is the revenue for Q3?",
        "results": [
            {
                "text": "Q3 revenue was $2.4M...",
                "score": 0.91,
                "source": "report.pdf",
                "page": 3
            },
            ...
        ]
    }
    """
    query: str           # echo back the original query
    results: list[dict]  # list of matching chunks with their similarity scores


# =============================================================================
# CHAT SCHEMAS
# =============================================================================
# Chat = full RAG pipeline in one request
# Question → embed → search Pinecone → build prompt → ask LLM → return answer

class ChatRequest(BaseModel):
    """
    Request body for the chat endpoint.
    
    Example request:
    { "question": "What were the key findings in the report?", "top_k": 5 }
    """
    question: str = Field(..., min_length=1)  # the user's question
    top_k: Optional[int] = 5                  # chunks to retrieve from Pinecone


class ChatResponse(BaseModel):
    """
    Response from the chat endpoint.
    Includes the LLM answer AND the source chunks it was based on.
    Sources let users verify the answer (reduces hallucination risk).
    
    Example response:
    {
        "answer": "The key findings were...",
        "sources": [
            {"text": "...", "score": 0.91, "source": "report.pdf", "page": 3},
            {"text": "...", "score": 0.87, "source": "report.pdf", "page": 5}
        ]
    }
    """
    answer: str          # the LLM generated answer
    sources: list[dict]  # the chunks used to generate the answer