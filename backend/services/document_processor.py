# io = built-in Python library for handling file data in memory
# (we don't save files to disk — we process them directly in RAM)
import io

# base64 = converts binary image data into text so we can send it to OpenAI API
import base64

# pandas = reads CSV and Excel files into a table structure
import pandas as pd

# Path = helps us extract file extension (.pdf, .csv, .png etc.)
from pathlib import Path

# our settings (OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP etc.)
from core.config import get_settings

settings = get_settings()


# =============================================================================
# STEP 1 — CHUNK TEXT
# =============================================================================
# After extracting text from a document, we split it into small pieces
# Reason: LLMs and vector DBs work better with short focused chunks
# than one giant wall of text

def chunk_text(text: str) -> list[str]:
    """
    Split a long text into smaller overlapping chunks.

    Why overlapping?
    Imagine a sentence split exactly at the boundary:
      chunk1: "The revenue for Q3 was"
      chunk2: "2.4 million dollars"
    
    Without overlap, searching "Q3 revenue" might miss the answer.
    With overlap, both chunks contain enough context to match.

    Example with CHUNK_SIZE=20, CHUNK_OVERLAP=5:
      text    = "Hello world this is a test sentence"
      chunk1  = "Hello world this is a"      (0 → 20)
      chunk2  = "is a test sentence"         (15 → 35)  ← starts 5 chars back
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + settings.CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:                          # skip empty chunks
            chunks.append(chunk)

        start += settings.CHUNK_SIZE - settings.CHUNK_OVERLAP  # move forward with overlap

    return chunks


# =============================================================================
# STEP 2 — IMAGE → TEXT  (Vision Model)
# =============================================================================
# PDFs often contain charts, diagrams, scanned pages — pure image with no text
# We send these images to GPT-4o which can SEE and describe them
# This is what makes the chatbot truly "multi-modal"

async def describe_image(image_bytes: bytes, media_type: str = "image/png") -> str:
    """
    Send an image to GPT-4o vision and get a text description back.

    How it works:
      1. Convert image bytes → base64 string (API only accepts text)
      2. Send to GPT-4o with instruction to describe it
      3. Get back a text description we can embed and search later

    media_type examples: "image/png", "image/jpeg", "image/webp"
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Convert raw image bytes to base64 encoded string
    # OpenAI API requires images in this format
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,  # gpt-4o (supports vision)
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        # Tell the API this message contains an image
                        "type": "image_url",
                        "image_url": {
                            # data URI format: data:<type>;base64,<data>
                            "url": f"data:{media_type};base64,{b64_image}",
                            "detail": "high",  # high = more tokens, better accuracy
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. "
                            "If it contains a chart or graph, extract all data values. "
                            "If it contains a table, extract all rows and columns. "
                            "If it contains text, transcribe it exactly."
                        ),
                    },
                ],
            }
        ],
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""


# =============================================================================
# STEP 3A — PDF → TEXT
# =============================================================================

async def extract_from_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Two scenarios per page:
      A) Page has text  → extract it directly (fast, free)
      B) Page is an image (scanned PDF, chart) → send to vision model (slower)

    Returns a list of page dicts:
    [
        {"text": "...", "page": 1, "source": "report.pdf", "content_type": "pdf"},
        {"text": "...", "page": 2, "source": "report.pdf", "content_type": "pdf"},
        ...
    ]
    """
    import fitz  # PyMuPDF — installed as "pymupdf"

    # Open the PDF from bytes (not from file path)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []

    for page_num, page in enumerate(doc, start=1):
        # Try to extract plain text from the page
        page_text = page.get_text().strip()

        # If text is very short, the page is probably a scanned image
        # Threshold: less than 50 characters = treat as image
        if len(page_text) < 50:
            # Render the page as a PNG image at 150 DPI
            pixmap = page.get_pixmap(dpi=150)
            image_bytes = pixmap.tobytes("png")

            # Send the image to GPT-4o vision for description
            page_text = await describe_image(image_bytes, "image/png")
            page_text = f"[Page {page_num} - Image Description]\n{page_text}"

        if page_text:
            pages.append({
                "text": page_text,
                "page": page_num,
                "source": filename,
                "content_type": "pdf",
            })

    return pages


# =============================================================================
# STEP 3B — IMAGE FILE → TEXT
# =============================================================================

async def extract_from_image(file_bytes: bytes, filename: str, media_type: str) -> list[dict]:
    """
    Handle standalone image uploads (PNG, JPG, WEBP).
    Directly sent to vision model for description.
    """
    description = await describe_image(file_bytes, media_type)

    return [{
        "text": description,
        "page": 1,
        "source": filename,
        "content_type": "image",
    }]


# =============================================================================
# STEP 3C — CSV / EXCEL → TEXT
# =============================================================================

async def extract_from_table(file_bytes: bytes, filename: str, extension: str) -> list[dict]:
    """
    Convert spreadsheet data into markdown text.

    Why markdown?
    It preserves the table structure in plain text so the LLM
    can read rows and columns properly.

    Example output:
    | Name  | Revenue | Quarter |
    |-------|---------|---------|
    | ACME  | $2.4M   | Q3      |
    """
    if extension == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:  # .xlsx or .xls
        df = pd.read_excel(io.BytesIO(file_bytes))

    # Convert DataFrame to markdown table string
    # requires: pip install tabulate
    markdown_table = df.to_markdown(index=False)

    return [{
        "text": markdown_table,
        "page": 1,
        "source": filename,
        "content_type": "table",
    }]


# =============================================================================
# MAIN FUNCTION — called by the upload router
# =============================================================================

async def process_document(file_bytes: bytes, filename: str, content_type: str) -> list[dict]:
    """
    Master function that:
      1. Detects file type
      2. Routes to the correct extractor
      3. Chunks all extracted text
      4. Returns ready-to-embed chunks

    Each returned chunk looks like:
    {
        "text": "Revenue for Q3 was 2.4 million...",
        "page": 3,
        "source": "report.pdf",
        "content_type": "pdf",
        "chunk_index": 0        ← which chunk of that page
    }
    """
    ext = Path(filename).suffix.lower()  # e.g. ".pdf", ".png", ".csv"
    raw_pages = []

    # Route to the correct extractor based on file extension
    if ext == ".pdf":
        raw_pages = await extract_from_pdf(file_bytes, filename)

    elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
        raw_pages = await extract_from_image(file_bytes, filename, content_type)

    elif ext in {".csv", ".xlsx", ".xls"}:
        raw_pages = await extract_from_table(file_bytes, filename, ext)

    else:
        # Fallback: treat as plain text file
        raw_pages = [{
            "text": file_bytes.decode("utf-8", errors="ignore"),
            "page": 1,
            "source": filename,
            "content_type": "text",
        }]

    # Now chunk every page's text and flatten into one list
    all_chunks = []

    for page_data in raw_pages:
        page_chunks = chunk_text(page_data["text"])

        for i, chunk in enumerate(page_chunks):
            all_chunks.append({
                **page_data,           # copy page, source, content_type
                "text": chunk,         # replace full text with this chunk
                "chunk_index": i,      # position of this chunk within the page
            })

    return all_chunks