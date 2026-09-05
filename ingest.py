import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

from src import config
from src import db
from src.embeddings import CloudflareTextEmbeddings, LocalCLIPEmbeddings

def run_ingestion(force_reingest: bool = False):
    """Run full or incremental ingestion pipeline for the apparel dataset."""
    print("=" * 60)
    print("Starting Multimodal Apparel Dataset Ingestion")
    print("=" * 60)

    # 1. Validate Database URL
    if db.is_neon_configured():
        print(f"\nTarget Database: NeonDB PostgreSQL + pgvector")
    else:
        print("\nℹ️ NOTICE: NEON_DATABASE_URL is not set in .env.")
        print("Ingesting into local catalog cache (SQLite + vectors).")
        print("Once you add your Neon connection URL to .env, run ingest.py again to populate NeonDB!\n")

    # 2. Initialize Database and pgvector extension
    print("\nStep 1/5: Initializing database schema...")
    db.init_db()

    # 3. Load Dataset
    print("\nStep 2/5: Loading products.jsonl...")
    if not config.PRODUCTS_JSONL.exists():
        print(f"❌ ERROR: Cannot find {config.PRODUCTS_JSONL}")
        sys.exit(1)

    with open(config.PRODUCTS_JSONL, "r", encoding="utf-8") as f:
        products_data = [json.loads(line) for line in f]
    print(f"Loaded {len(products_data)} products from {config.PRODUCTS_JSONL}")

    # Check for existing records to support fast incremental ingestion
    existing_skus = set(db.get_existing_skus())
    if not force_reingest and existing_skus:
        products_to_ingest = [p for p in products_data if str(p["sku"]) not in existing_skus]
        if not products_to_ingest:
            print(f"\n✅ All {len(products_data)} products are already up to date in the database!")
            print("Tip: Add new lines to rag_dataset/products.jsonl to ingest new products,")
            print("     or run 'python ingest.py --force' to re-embed all items.\n")
            return
        print(f"\n⚡ Incremental Ingestion: Found {len(products_to_ingest)} new product(s) to embed and insert.")
    else:
        products_to_ingest = products_data
        print(f"\nFull Ingestion: Processing all {len(products_to_ingest)} products.")

    # 4. Generate Text Documents and Text Embeddings via Cloudflare
    print(f"\nStep 3/5: Generating text embeddings via Cloudflare Workers AI ({config.CLOUDFLARE_TEXT_EMBED_MODEL})...")
    text_embedder = CloudflareTextEmbeddings()
    
    text_documents = []
    for p in products_to_ingest:
        # Atomic composite text representation
        doc_text = (
            f"Title: {p['title']}. "
            f"Category: {p['category']}. "
            f"Gender: {p['gender']}. "
            f"Price: INR {p['price_inr']}. "
            f"Tagline: {p.get('tagline', '')}. "
            f"Description: {p.get('description', '')}."
        )
        text_documents.append(doc_text)

    text_embeddings = text_embedder.embed_documents(text_documents)
    print(f"Successfully generated {len(text_embeddings)} text embeddings (dim: {len(text_embeddings[0])}).")

    # 5. Generate CLIP Image Embeddings locally
    print(f"\nStep 4/5: Generating CLIP image embeddings locally ({config.CLIP_MODEL_NAME})...")
    clip_embedder = LocalCLIPEmbeddings()

    image_paths = []
    for p in products_to_ingest:
        # Ensure path to image
        img_rel = p.get("image_file", "")
        img_full_path = config.DATASET_DIR / img_rel
        if not img_full_path.exists():
            # Fallback check if images/ is in img_rel
            img_full_path = config.IMAGES_DIR / Path(img_rel).name
        
        if not img_full_path.exists():
            raise FileNotFoundError(f"Missing image file for SKU {p['sku']}: {img_full_path}")
        image_paths.append(str(img_full_path))

    # Batch compute CLIP image vectors
    image_embeddings = clip_embedder.embed_images_batch(image_paths, batch_size=16)
    print(f"Successfully generated {len(image_embeddings)} CLIP image embeddings (dim: {len(image_embeddings[0])}).")

    # 6. Combine and Upsert to NeonDB
    print("\nStep 5/5: Upserting records into NeonDB pgvector...")
    records_to_insert = []
    for p, text_vec, img_vec, img_path in zip(products_to_ingest, text_embeddings, image_embeddings, image_paths):
        # Normalize relative image path for UI rendering
        rel_path = os.path.relpath(img_path, config.BASE_DIR).replace("\\", "/")
        records_to_insert.append({
            "sku": str(p["sku"]),
            "title": p["title"],
            "category": p["category"],
            "gender": p["gender"],
            "price_inr": int(p["price_inr"]),
            "description": p.get("description", ""),
            "tagline": p.get("tagline", ""),
            "image_file": rel_path,
            "text_embedding": text_vec,
            "image_embedding": img_vec,
        })

    db.upsert_products(records_to_insert)

    # 7. Print Catalog Summary
    stats = db.get_catalog_stats()
    print("\n" + "=" * 60)
    print("Ingestion Complete! Catalog Summary in NeonDB:")
    print(f"- Total Products: {stats['total_products']}")
    print(f"- Categories: {stats['categories']}")
    print(f"- Price Range: ₹{stats['min_price']} - ₹{stats['max_price']} (Avg: ₹{stats['avg_price']})")
    print("=" * 60)

if __name__ == "__main__":
    force = "--force" in sys.argv or "-f" in sys.argv
    run_ingestion(force_reingest=force)


