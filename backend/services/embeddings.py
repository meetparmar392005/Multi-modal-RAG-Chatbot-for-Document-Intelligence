# OpenAI's async client — "async" means it won't block other requests
# while waiting for the API response
from openai import AsyncOpenAI

# our settings (OPENAI_API_KEY, EMBEDDING_MODEL etc.)
from core.config import get_settings

settings = get_settings()

# Create ONE shared OpenAI client for the whole app
# (creating a new client per request is wasteful)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


# =============================================================================
# WHAT IS AN EMBEDDING?
# =============================================================================
# An embedding is a list of numbers (vector) that represents the MEANING of text
#
# Example:
#   "dog"  → [0.21, -0.45, 0.89, ...]   (1536 numbers)
#   "cat"  → [0.19, -0.41, 0.91, ...]   (1536 numbers — similar to dog!)
#   "car"  → [0.87,  0.12, -0.34, ...]  (very different numbers)
#
# Similar meanings → similar numbers → close together in vector space
# This is how semantic search works — not matching keywords but matching MEANING


# =============================================================================
# ENCODER — Text → Vector
# =============================================================================

async def encode(text: str) -> list[float]:
    """
    Convert a single piece of text into a vector (list of numbers).
    
    This is called:
      - During upload   → embed each chunk → store in Pinecone
      - During chat     → embed the user's question → search Pinecone

    Example:
      input:  "What was the Q3 revenue?"
      output: [0.021, -0.045, 0.089, ...]  ← 1536 numbers
    """
    response = await client.embeddings.create(
        model=settings.EMBEDDING_MODEL,   # text-embedding-3-small
        input=text,                        # the text to embed
    )

    # The API returns an object — we extract just the list of floats
    return response.data[0].embedding


async def encode_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts in ONE API call — much faster than one by one.
    
    Used during upload to embed all chunks together.

    Example:
      input:  ["chunk 1 text", "chunk 2 text", "chunk 3 text"]
      output: [[0.02, ...], [0.11, ...], [0.08, ...]]  ← one vector per chunk
    
    Why batch?
      Embedding 100 chunks one-by-one = 100 API calls
      Embedding 100 chunks in a batch = 1 API call  ← much faster & cheaper
    """
    response = await client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,   # pass the whole list at once
    )

    # Sort by index to guarantee order matches the input list
    # (API may return them out of order)
    sorted_data = sorted(response.data, key=lambda x: x.index)

    return [item.embedding for item in sorted_data]


# =============================================================================
# DECODER — Query Text → Search Vector DB → Return similar texts
# =============================================================================
# "Decode" here means: given a question, find the most similar stored chunks
# This is semantic search — the core of RAG retrieval

async def decode(query: str, vector_store, top_k: int = None) -> list[dict]:
    """
    Convert a query to a vector, then search the vector store for similar chunks.

    Steps:
      1. Embed the query text into a vector
      2. Send that vector to Pinecone
      3. Pinecone returns the top_k most similar stored vectors
      4. We return the matching text chunks with their similarity scores

    Example:
      input:  "What was the Q3 revenue?"
      output: [
                {"text": "Q3 revenue was $2.4M...", "score": 0.92, "source": "report.pdf", "page": 3},
                {"text": "Revenue grew 12% in Q3...", "score": 0.87, "source": "report.pdf", "page": 5},
              ]

    score = similarity between 0.0 and 1.0
      1.0 = perfect match
      0.0 = completely unrelated
    """
    if top_k is None:
        top_k = settings.TOP_K_RESULTS

    # Step 1 — embed the query using the same model used during upload
    # IMPORTANT: must use the SAME model, otherwise vectors are incompatible
    query_vector = await encode(query)

    # Step 2 — search Pinecone with the query vector
    # vector_store is passed in from the router (dependency injection)
    results = await vector_store.search(query_vector, top_k=top_k)

    return results


# =============================================================================
# UTILITY — get embedding dimensions
# =============================================================================

async def get_dimensions() -> int:
    """
    Returns how many numbers are in each embedding vector.
    
    text-embedding-3-small → 1536 dimensions
    text-embedding-3-large → 3072 dimensions

    This is needed when creating a Pinecone index
    (Pinecone needs to know the vector size upfront)
    """
    test_vector = await encode("test")
    return len(test_vector)