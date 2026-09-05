import os
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pgvector.psycopg2 import register_vector
from src import config

def is_neon_configured() -> bool:
    """Check if Neon PostgreSQL URL is configured."""
    return bool(config.NEON_DATABASE_URL and config.NEON_DATABASE_URL.startswith("postgres"))

def get_connection():
    """Establish connection to NeonDB PostgreSQL and register pgvector extension."""
    if not is_neon_configured():
        raise ValueError("NEON_DATABASE_URL is not set. Please set it in your .env file.")
    conn = psycopg2.connect(config.NEON_DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        register_vector(conn)
    except psycopg2.ProgrammingError:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
        register_vector(conn)
    return conn

def init_db():
    """Initialize schema in NeonDB PostgreSQL with pgvector and HNSW indexes."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS products (
                    sku TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    gender TEXT NOT NULL,
                    price_inr INTEGER NOT NULL,
                    description TEXT,
                    tagline TEXT,
                    image_file TEXT,
                    text_embedding vector({config.TEXT_EMBEDDING_DIM}),
                    image_embedding vector({config.IMAGE_EMBEDDING_DIM}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_products_gender ON products(gender);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_products_price ON products(price_inr);")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_products_text_hnsw 
                ON products USING hnsw (text_embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_products_image_hnsw 
                ON products USING hnsw (image_embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            conn.commit()
            print("NeonDB initialized with pgvector and HNSW indexes.")
    finally:
        conn.close()

def upsert_products(products: List[Dict[str, Any]]):
    """Upsert products into NeonDB pgvector."""
    if not products:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            insert_query = """
                INSERT INTO products (
                    sku, title, category, gender, price_inr, description, tagline, 
                    image_file, text_embedding, image_embedding
                ) VALUES %s
                ON CONFLICT (sku) DO UPDATE SET
                    title = EXCLUDED.title,
                    category = EXCLUDED.category,
                    gender = EXCLUDED.gender,
                    price_inr = EXCLUDED.price_inr,
                    description = EXCLUDED.description,
                    tagline = EXCLUDED.tagline,
                    image_file = EXCLUDED.image_file,
                    text_embedding = EXCLUDED.text_embedding,
                    image_embedding = EXCLUDED.image_embedding;
            """
            data_tuples = [
                (
                    p["sku"],
                    p["title"],
                    p["category"],
                    p["gender"],
                    p["price_inr"],
                    p.get("description", ""),
                    p.get("tagline", ""),
                    p.get("image_file", ""),
                    p.get("text_embedding"),
                    p.get("image_embedding"),
                )
                for p in products
            ]
            execute_values(cur, insert_query, data_tuples)
            conn.commit()
            print(f"Upserted {len(products)} products into NeonDB pgvector.")
    finally:
        conn.close()

def _build_where_clause(filters: Optional[Dict[str, Any]] = None, placeholder: str = "%s"):
    """Build SQL WHERE conditions and params from structured filters."""
    conditions = []
    params = []
    if not filters:
        return "", params

    if filters.get("category"):
        conditions.append(f"category LIKE {placeholder}")
        params.append(f"%{filters['category']}%")

    if filters.get("gender"):
        conditions.append(f"gender = {placeholder}")
        params.append(f"{filters['gender']}")

    if filters.get("max_price") is not None:
        conditions.append(f"price_inr <= {placeholder}")
        params.append(filters["max_price"])

    if filters.get("min_price") is not None:
        conditions.append(f"price_inr >= {placeholder}")
        params.append(filters["min_price"])

    where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where_sql, params

def search_by_text_embedding(
    vector: List[float],
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Hybrid search: filter by metadata in SQL WHERE clause, then rank by cosine distance."""
    conn = get_connection()
    try:
        where_sql, params = _build_where_clause(filters, "%s")
        query = f"""
            SELECT sku, title, category, gender, price_inr, description, tagline, image_file,
                   ROUND((1 - (text_embedding <=> %s::vector))::numeric, 4) AS similarity
            FROM products
            {where_sql}
            ORDER BY text_embedding <=> %s::vector ASC
            LIMIT %s;
        """
        full_params = [vector] + params + [vector, limit]
        with conn.cursor() as cur:
            cur.execute(query, full_params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

def search_by_image_embedding(
    vector: List[float],
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Visual search: filter by metadata then rank by image_embedding cosine similarity."""
    conn = get_connection()
    try:
        where_sql, params = _build_where_clause(filters, "%s")
        query = f"""
            SELECT sku, title, category, gender, price_inr, description, tagline, image_file,
                   ROUND((1 - (image_embedding <=> %s::vector))::numeric, 4) AS similarity
            FROM products
            {where_sql}
            ORDER BY image_embedding <=> %s::vector ASC
            LIMIT %s;
        """
        full_params = [vector] + params + [vector, limit]
        with conn.cursor() as cur:
            cur.execute(query, full_params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

def search_metadata_only(
    filters: Dict[str, Any],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Direct SQL search bypassing vector calculation when exact metadata match is requested."""
    conn = get_connection()
    try:
        where_sql, params = _build_where_clause(filters, "%s")
        query = f"""
            SELECT sku, title, category, gender, price_inr, description, tagline, image_file,
                   1.0 AS similarity
            FROM products
            {where_sql}
            ORDER BY price_inr ASC
            LIMIT %s;
        """
        params.append(limit)
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

def get_catalog_stats() -> Dict[str, Any]:
    """Retrieve catalog statistics from NeonDB."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM products;")
            total = cur.fetchone()["total"]
            cur.execute("SELECT category, COUNT(*) AS count FROM products GROUP BY category ORDER BY count DESC;")
            category_counts = {r["category"]: r["count"] for r in cur.fetchall()}
            cur.execute("SELECT MIN(price_inr) AS min_p, MAX(price_inr) AS max_p, AVG(price_inr) AS avg_p FROM products;")
            price_row = cur.fetchone()
            return {
                "engine": "NeonDB PostgreSQL (pgvector)",
                "total_products": total,
                "categories": category_counts,
                "min_price": price_row["min_p"] if price_row else 0,
                "max_price": price_row["max_p"] if price_row else 0,
                "avg_price": round(price_row["avg_p"], 2) if price_row and price_row["avg_p"] else 0,
            }
    finally:
        conn.close()

def get_existing_skus() -> List[str]:
    """Retrieve all product SKUs currently in NeonDB."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT sku FROM products;")
            return [r["sku"] for r in cur.fetchall()]
    finally:
        conn.close()
