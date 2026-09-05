# PROSEARCH 🔍

An intelligent Multimodal Retrieval-Augmented Generation (RAG) system for fashion and apparel product discovery, featuring:
- **Cloudflare Workers AI**: Fast Text Embeddings (`@cf/baai/bge-base-en-v1.5`), LLM Generation (`@cf/meta/llama-3.1-8b-instruct-fp8`), and Safety Guardrails (`@cf/meta/llama-guard-3-8b`).
- **Local CLIP Vision Embeddings**: 512-dim multimodal vectors (`openai/clip-vit-base-patch32`) for visual similarity matching and cross-modal search.
- **NeonDB PostgreSQL + pgvector**: Unified database for structured metadata pre-filtering (SQL B-tree indexes) and approximate nearest-neighbor vector search (HNSW index).
- **Streamlit Web UI**: Interactive dashboard featuring natural language search, visual image upload search, direct SQL catalog filtering, and rich product cards.

---

## 🌟 Key Features

1. **Dual Embeddings on a Single Table**:
   - `text_embedding` (768-dim): dense representation of title, category, gender, price, tagline, and description.
   - `image_embedding` (512-dim): normalized CLIP visual representation of the product image.
2. **Fast Metadata Pre-Filtering (Bypassing Vector DB Overhead)**:
   - When users specify explicit filters (e.g., *"Men topwear under ₹1500"*), the query engine extracts structured attributes (`gender='Men'`, `category='Topwear'`, `price_inr <= 1500`) and executes fast SQL queries directly on indexed columns before or instead of vector search.
3. **Multimodal Search Capabilities**:
   - **Text Query**: Natural language fashion search with personal AI stylist advice.
   - **Visual Search**: Upload a photo or select a sample image to find visually similar items.
   - **Cross-Modal Search**: Natural language description mapped directly against image vectors in CLIP space.
   - **Direct Catalog Filter**: Instant SQL queries bypassing vector compute entirely.
4. **End-to-End Guardrails**:
   - **Input Safety**: Cloudflare Llama Guard 3 8B + prompt injection defense.
   - **Output Grounding**: Verifies all cited SKUs against retrieved products to eliminate hallucinations.

---

## 📁 Project Structure

```
Product_Search_RAG/
├── .env                        # Environment credentials (Cloudflare & NeonDB)
├── .env.example                # Template for environment variables
├── requirements.txt            # Project dependencies
├── ingest.py                   # Data ingestion script into NeonDB
├── app.py                      # Streamlit interactive UI
├── src/
│   ├── __init__.py
│   ├── config.py               # Central configuration & paths
│   ├── db.py                   # NeonDB pgvector schema, HNSW indexes & queries
│   ├── embeddings.py           # Cloudflare Text Embeddings & Local CLIP Embeddings
│   ├── guardrails.py           # Safety Guardrail, Intent/Filter Extractor & Grounding Check
│   └── rag.py                  # RAG pipeline orchestration & Cloudflare LLM
├── rag_dataset/                # Provided catalog dataset
│   ├── images/                 # 150 product images (.jpg)
│   ├── products.csv
│   └── products.jsonl
└── tests/
    └── test_pipeline.py        # Automated test suite
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
The project uses Python 3.11:
```powershell
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Configure Credentials (`.env`)
Ensure your `.env` file contains your Cloudflare credentials and your NeonDB PostgreSQL connection string:
```env
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
CLOUDFLARE_TEXT_EMBED_MODEL=@cf/baai/bge-base-en-v1.5
CLOUDFLARE_LLM_MODEL=@cf/meta/llama-3.1-8b-instruct-fp8
CLOUDFLARE_GUARD_MODEL=@cf/meta/llama-guard-3-8b

# NeonDB PostgreSQL Connection String
NEON_DATABASE_URL=postgresql://<user>:<password>@<neon-hostname>/<dbname>?sslmode=require

CLIP_MODEL_NAME=openai/clip-vit-base-patch32
```

### 3. Run Ingestion Pipeline
To initialize the NeonDB database with `pgvector`, build the HNSW indexes, compute text and image embeddings, and upsert all 150 products:
```powershell
.venv\Scripts\python.exe ingest.py
```

### 4. Run Automated Tests
```powershell
.venv\Scripts\python.exe -m unittest tests/test_pipeline.py
```

### 5. Launch the Streamlit UI
```powershell
.venv\Scripts\streamlit.exe run app.py
```
---

## 🌐 Deploying to Render
1. **Build Command**: `pip install -r requirements.txt && python download_model.py`
2. **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. **Environment Variables**: Add your `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `NEON_DATABASE_URL`, and `PYTHON_VERSION=3.11.9`.

---

## 💓 Health Check & Uptime Monitoring
To prevent Render free/inactivity sleep, point **UptimeRobot** (HTTP GET every 5 minutes) to:
- `https://<your-app-url>/health`
- OR native Streamlit health check: `https://<your-app-url>/_stcore/health`



