import os
from dotenv import load_dotenv

# load variables from .env into the environment
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY. Add it to your .env file.")

# where we'll store PDFs and the vector index
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

# name of the collection inside chroma
COLLECTION_NAME = "documents"

# how big each text chunk should be, and how much they overlap
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# claude model to use for answering
CLAUDE_MODEL = "claude-sonnet-4-5"