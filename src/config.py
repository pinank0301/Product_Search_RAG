import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Use Windows native system SSL certificate store (handles corporate proxy/Zscaler/Capgemini root certs)
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# Ensure standard HTTP downloads for Hugging Face
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# Cloudflare Configuration
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_TEXT_EMBED_MODEL = os.getenv("CLOUDFLARE_TEXT_EMBED_MODEL", "@cf/baai/bge-base-en-v1.5")
CLOUDFLARE_LLM_MODEL = os.getenv("CLOUDFLARE_LLM_MODEL", "@cf/meta/llama-3.1-8b-instruct-fp8")
CLOUDFLARE_GUARD_MODEL = os.getenv("CLOUDFLARE_GUARD_MODEL", "@cf/meta/llama-guard-3-8b")

# NeonDB / PostgreSQL Connection
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL", "")

# Vision Model Configuration
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# Vector Dimensions
# @cf/baai/bge-base-en-v1.5 is 768; @cf/baai/bge-small-en-v1.5 is 384
TEXT_EMBEDDING_DIM = 768 if "base" in CLOUDFLARE_TEXT_EMBED_MODEL else (384 if "small" in CLOUDFLARE_TEXT_EMBED_MODEL else 1024)
IMAGE_EMBEDDING_DIM = 512

# Dataset Paths
DATASET_DIR = BASE_DIR / "rag_dataset"
PRODUCTS_JSONL = DATASET_DIR / "products.jsonl"
PRODUCTS_CSV = DATASET_DIR / "products.csv"
IMAGES_DIR = DATASET_DIR / "images"

