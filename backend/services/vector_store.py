# uuid = generates a unique ID for each chunk we store in Pinecone
# every vector in Pinecone needs a unique string ID
import uuid

# our settings (PINECONE_API_KEY, PINECONE_INDEX_NAME, TOP_K_RESULTS)
from core.config import get_settings

settings = get_settings()


# =============================================================================
# WHAT IS A VECTOR DATABASE?
# =============================================================================
# A normal database searches by exact match  → "find where name = 'John'"
# A vector database searches by similarity   → "find vectors CLOSE to this vector"
#
# Pinecone stores each chunk as:
#   {
#       "id":       "some-unique-id",        ← required, must be unique
#       "values":   [0.021, -0.045, ...],    ← the embedding vector
#       "metadata": {                         ← extra info stored alongside
#           "text":    "Q3 revenue was...",
#           "source":  "report.pdf",
#           "page":    3,
#       }
#   }
#
# When we search, Pinecone finds the vectors mathematically closest
# to our query vector and returns their metadata (the original text)


# =============================================================================
# VECTOR STORE CLASS
# =============================================================================

class VectorStore:
    """
    Wraps Pinecone so the rest of the app never talks to Pinecone directly.
    If we ever switch from Pinecone to Weaviate, we only change THIS file.
    """

    def __init__(self):
        # index = the Pinecone "table" where all our vectors live
        # set to None at start — connected in connect()
        self.index = None

    async def connect(self):
        """
        Connect to Pinecone and get our index ready.
        Called once when the app starts up.

        Two scenarios:
          A) Index already exists → just connect to it
          B) Index doesn't exist  → create it first, then connect
        """
        from pinecone import Pinecone, ServerlessSpec

        # Create the Pinecone client using our API key
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        # Check if our index already exists
        existing_indexes = [i.name for i in pc.list_indexes()]

        if settings.PINECONE_INDEX_NAME not in existing_indexes:
            # Index doesn't exist yet — create it
            # dimension=1536 must match our embedding model output size
            # text-embedding-3-small always produces 1536-dimensional vectors
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",          # cosine similarity = best for text
                spec=ServerlessSpec(
                    cloud="aws",          # Pinecone free tier runs on AWS
                    region="us-east-1",
                ),
            )
            print(f"Created new Pinecone index: {settings.PINECONE_INDEX_NAME}")

        # Connect to the index (whether it already existed or we just created it)
        self.index = pc.Index(settings.PINECONE_INDEX_NAME)
        print(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")

    # =========================================================================
    # STORE — save embeddings into Pinecone
    # =========================================================================

    async def store(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """
        Save a list of text chunks and their vectors into Pinecone.

        chunks     = list of {text, page, source, content_type, chunk_index}
        embeddings = list of vectors, same order as chunks

        Example:
          chunks[0]     = {"text": "Q3 revenue was...", "source": "report.pdf", "page": 3}
          embeddings[0] = [0.021, -0.045, 0.089, ...]

        Pinecone requires data in "upsert" format:
          upsert = insert if new, update if ID already exists
        """
        if not self.index:
            raise RuntimeError("VectorStore not connected. Call connect() first.")

        # Build the list of vectors to upsert
        vectors_to_upsert = []

        for chunk, embedding in zip(chunks, embeddings):
            vectors_to_upsert.append({
                # unique ID for this chunk — uuid4 generates a random unique string
                "id": str(uuid.uuid4()),

                # the actual embedding vector
                "values": embedding,

                # metadata = all the extra info we want back when searching
                # this is how we get the original text back from Pinecone
                "metadata": {
                    "text":         chunk["text"],
                    "source":       chunk["source"],
                    "page":         chunk.get("page", 1),
                    "content_type": chunk.get("content_type", "text"),
                    "chunk_index":  chunk.get("chunk_index", 0),
                },
            })

        # Pinecone recommends upserting in batches of 100
        # Sending thousands at once can cause timeouts
        batch_size = 100

        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            self.index.upsert(vectors=batch)

        return len(vectors_to_upsert)   # return how many were stored

    # =========================================================================
    # SEARCH — find most similar chunks to a query vector
    # =========================================================================

    async def search(self, query_vector: list[float], top_k: int = None) -> list[dict]:
        """
        Search Pinecone for the most similar vectors to the query.

        query_vector = the embedded version of the user's question
        top_k        = how many results to return

        Returns list of matching chunks:
        [
            {
                "text":   "Q3 revenue was $2.4M...",
                "score":  0.92,               ← similarity (1.0 = perfect)
                "source": "report.pdf",
                "page":   3,
            },
            ...
        ]

        Score meaning:
          0.9 - 1.0 → very relevant
          0.7 - 0.9 → probably relevant
          below 0.7 → likely not relevant
        """
        if not self.index:
            raise RuntimeError("VectorStore not connected. Call connect() first.")

        if top_k is None:
            top_k = settings.TOP_K_RESULTS

        # Query Pinecone
        # include_metadata=True → return the text and source info alongside scores
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        # Format results into clean dicts
        results = []

        for match in response.matches:
            results.append({
                "text":         match.metadata.get("text", ""),
                "score":        round(match.score, 4),   # similarity score
                "source":       match.metadata.get("source", ""),
                "page":         match.metadata.get("page", 1),
                "content_type": match.metadata.get("content_type", ""),
            })

        return results

    # =========================================================================
    # DELETE — remove all vectors for a specific file
    # =========================================================================

    async def delete_by_source(self, filename: str):
        """
        Delete all stored chunks that came from a specific file.
        Useful when a user re-uploads an updated version of a document.
        """
        if not self.index:
            raise RuntimeError("VectorStore not connected. Call connect() first.")

        # Pinecone lets us delete by metadata filter
        self.index.delete(
            filter={"source": {"$eq": filename}}
        )
        print(f"Deleted all vectors for source: {filename}")

    # =========================================================================
    # STATS — useful for debugging
    # =========================================================================

    async def get_stats(self) -> dict:
        """
        Returns info about the Pinecone index.
        Useful to verify vectors were stored correctly.

        Example output:
        {
            "total_vectors": 142,
            "dimensions": 1536,
            "index_name": "rag-index"
        }
        """
        if not self.index:
            raise RuntimeError("VectorStore not connected. Call connect() first.")

        stats = self.index.describe_index_stats()

        return {
            "total_vectors": stats.total_vector_count,
            "dimensions":    stats.dimension,
            "index_name":    settings.PINECONE_INDEX_NAME,
        }


# =============================================================================
# SHARED INSTANCE
# =============================================================================
# Create ONE VectorStore instance for the whole app
# All routers import this same object — no duplicate connections

vector_store = VectorStore()