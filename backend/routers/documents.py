# APIRouter = creates a mini FastAPI app we attach to main.py later
# File, UploadFile = FastAPI's built-in file upload handling
# HTTPException = returns proper HTTP error responses (400, 500 etc.)
# Depends = dependency injection — shares objects across requests
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

# our response schema (the shape of what we return)
from models.schemas import UploadResponse, SearchRequest, SearchResponse

# the three services this router uses
from services.document_processor import process_document
from services.embeddings import encode_batch, decode
from services.vector_store import vector_store

# our settings
from core.config import get_settings

settings = get_settings()

# Create the router with a prefix so all routes start with /documents
# tag = groups these routes together in the auto-generated API docs
router = APIRouter(prefix="/documents", tags=["Documents"])


# =============================================================================
# ALLOWED FILE TYPES
# =============================================================================

ALLOWED_TYPES = {
    "application/pdf",       # PDF files
    "image/png",             # PNG images
    "image/jpeg",            # JPG images
    "image/webp",            # WEBP images
    "text/csv",              # CSV spreadsheets
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",  # .xls
}

# Max file size = 20MB (in bytes)
MAX_FILE_SIZE = 20 * 1024 * 1024


# =============================================================================
# POST /documents/upload
# =============================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and process it into searchable embeddings.

    Full flow:
      1. Validate file type and size
      2. Extract text from the file (PDF/image/table)
      3. Split text into chunks
      4. Embed all chunks into vectors
      5. Store vectors in Pinecone
      6. Return how many chunks were created and stored

    Accepts: PDF, PNG, JPG, WEBP, CSV, XLSX
    """

    # ── Step 1: Validate file type ────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file.content_type}' not supported. "
                   f"Allowed: PDF, PNG, JPG, WEBP, CSV, XLSX"
        )

    # ── Step 2: Read file bytes into memory ───────────────────────────────────
    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 20MB."
        )

    # ── Step 3: Extract text and chunk it ─────────────────────────────────────
    # process_document routes to the right extractor based on file type
    # returns list of {text, page, source, content_type, chunk_index}
    try:
        chunks = await process_document(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this file."
        )

    # ── Step 4: Embed all chunks in one batch API call ────────────────────────
    # extract just the text from each chunk for embedding
    texts = [chunk["text"] for chunk in chunks]

    try:
        embeddings = await encode_batch(texts)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )

    # ── Step 5: Store vectors + metadata in Pinecone ──────────────────────────
    try:
        stored_count = await vector_store.store(chunks, embeddings)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store embeddings: {str(e)}"
        )

    # ── Step 6: Return success response ───────────────────────────────────────
    return UploadResponse(
        file_name=file.filename,
        chunks_created=len(chunks),
        embeddings_stored=stored_count,
        message=f"'{file.filename}' processed successfully.",
    )


# =============================================================================
# POST /documents/search
# =============================================================================

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Semantic search across all stored documents.

    Flow:
      1. Embed the query text into a vector
      2. Search Pinecone for the most similar vectors
      3. Return the matching text chunks with scores

    Different from /chat — this returns raw chunks, no LLM answer.
    Useful for testing retrieval quality before wiring up the LLM.

    Example request:
      { "query": "what was the Q3 revenue?", "top_k": 5 }

    Example response:
      {
        "query": "what was the Q3 revenue?",
        "results": [
          {"text": "Q3 revenue was $2.4M...", "score": 0.92, "source": "report.pdf", "page": 3},
          ...
        ]
      }
    """
    try:
        results = await decode(
            query=request.query,
            vector_store=vector_store,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

    return SearchResponse(
        query=request.query,
        results=results,
    )
# =============================================================================
# GET /documents/stats
# =============================================================================

@router.get("/stats")
async def get_stats():
    """
    Returns info about the Pinecone index.
    Use this to verify uploads worked correctly.

    Example response:
    {
        "total_vectors": 142,
        "dimensions": 1536,
        "index_name": "rag-index"
    }
    """
    try:
        stats = await vector_store.get_stats()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )

    return stats

# =============================================================================
# DELETE /documents/{filename}
# =============================================================================

@router.delete("/{filename}")
async def delete_document(filename: str):
    """
    Delete all stored chunks for a specific file from Pinecone.
    Call this before re-uploading an updated version of the same file.

    Example:
      DELETE /documents/report.pdf
      → removes all vectors where source = "report.pdf"
    """
    try:
        await vector_store.delete_by_source(filename)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

    return {"message": f"All chunks for '{filename}' deleted successfully."}


