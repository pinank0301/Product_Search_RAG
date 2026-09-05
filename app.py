import os
import sys
import json
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List

from src import config
from src import db
from src.rag import ProductRAGPipeline

# Attach /health endpoint to Streamlit's underlying Tornado server
def _register_health_endpoint():
    try:
        from streamlit.web.server.server import Server
        import tornado.web

        class HealthHandler(tornado.web.RequestHandler):
            def get(self):
                self.set_header("Content-Type", "application/json")
                self.set_header("Cache-Control", "no-cache")
                self.write(json.dumps({"status": "ok", "service": "Product_Search_RAG"}))

            def head(self):
                self.set_status(200)

        server = Server.get_current()
        if server and hasattr(server, "_app") and server._app:
            server._app.add_handlers(r".*", [(r"/health", HealthHandler)])
    except Exception:
        pass

_register_health_endpoint()

# Fast-path for query parameter health checks (e.g. /?health=1 or /?route=health)
if st.query_params.get("health") is not None or st.query_params.get("route") == "health":
    st.json({"status": "ok", "service": "Product_Search_RAG"})
    st.stop()


# Page Configuration
st.set_page_config(
    page_title="PROSEARCH - Fashion & Apparel Discovery",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Polished, Theme-Adaptive Styling (Works seamlessly in both Dark and Light mode)
st.markdown("""
<style>
    .brand-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .brand-subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.8rem;
    }
    .price-text {
        font-size: 1.35rem;
        font-weight: 700;
        color: #10B981;
        margin: 6px 0;
    }
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        background-color: rgba(99, 102, 241, 0.12);
        color: #6366F1;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .pill-green {
        background-color: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .pill-amber {
        background-color: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Cache the RAG pipeline instance."""
    return ProductRAGPipeline()


def render_product_card(p: Dict[str, Any], show_similarity: bool = True):
    """Render a single product card inside a native theme-adaptive container."""
    with st.container(border=True):
        col_img, col_info = st.columns([1, 1.4], gap="medium")

        with col_img:
            img_rel = p.get("image_file", "")
            img_path = config.BASE_DIR / img_rel
            if not img_path.exists():
                img_path = config.IMAGES_DIR / Path(img_rel).name

            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.info("Image not available")

        with col_info:
            st.subheader(p["title"])

            # Pills for SKU, Category, Gender, and Match
            pills_html = (
                f'<span class="pill">SKU {p["sku"]}</span>'
                f'<span class="pill pill-amber">{p["category"]} · {p["gender"]}</span>'
            )
            if show_similarity and "similarity" in p:
                sim_val = p["similarity"]
                pct = f"{float(sim_val) * 100:.0f}%" if isinstance(sim_val, (int, float)) else str(sim_val)
                pills_html += f'<span class="pill pill-green">{pct} Match</span>'

            st.markdown(pills_html, unsafe_allow_html=True)
            st.markdown(f'<div class="price-text">₹{p["price_inr"]:,}</div>', unsafe_allow_html=True)

            if p.get("tagline"):
                st.caption(f"*{p['tagline']}*")

            if p.get("description"):
                st.write(p["description"])


def main():
    # Brand Header
    st.markdown('<div class="brand-title">🔍 PROSEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Smart AI Fashion & Product Search Assistant</div>', unsafe_allow_html=True)

    # Sidebar Navigation & Filters
    st.sidebar.title("Navigation & Filters")

    # Catalog status indicator
    db_connected = False
    try:
        stats = db.get_catalog_stats()
        db_connected = True
        st.sidebar.success(f"Catalog Active: {stats.get('total_products', 0)} Products")
    except Exception:
        st.sidebar.warning("Connecting to catalog...")

    with st.sidebar.expander("📦 Sync New Products"):
        st.caption("Add items to `products.jsonl` & `images/`, then click:")
        if st.button("🔄 Sync Catalog Now", use_container_width=True):
            with st.spinner("Checking & embedding new products..."):
                from ingest import run_ingestion
                run_ingestion()
            st.success("Sync complete!")
            st.rerun()

    # Search Mode Selector
    search_mode = st.sidebar.radio(
        "Search Mode:",
        [
            "Natural Language Search",
            "Visual Search (Image Match)"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Preferences")

    auto_extract = st.sidebar.checkbox("Auto-detect filters from query", value=True)
    selected_gender = st.sidebar.selectbox("Gender", ["All", "Men", "Women"])
    selected_cat = st.sidebar.selectbox(
        "Category",
        ["All", "Topwear", "Bottomwear", "Dress", "Saree", "Apparel Set"]
    )
    max_price = st.sidebar.slider("Maximum Price (₹)", min_value=500, max_value=7000, value=7000, step=100)
    top_k = st.sidebar.slider("Results to Show", min_value=2, max_value=12, value=4)

    manual_filters = {}
    if selected_gender != "All":
        manual_filters["gender"] = selected_gender
    if selected_cat != "All":
        manual_filters["category"] = selected_cat
    if max_price < 7000:
        manual_filters["max_price"] = max_price

    pipeline = get_pipeline()

    # MODE 1: Natural Language Search
    if search_mode == "Natural Language Search":
        col_q, col_sample = st.columns([3, 1])
        with col_sample:
            sample_query = st.selectbox(
                "Quick Suggestions",
                [
                    "",
                    "Men stylish topwear under 1500",
                    "Casual bottomwear pants for men",
                    "Elegant saree under 3000",
                    "Comfortable summer dress for women",
                    "Apparel sets under 2500"
                ]
            )

        with col_q:
            query_input = st.text_input(
                "What are you looking for today?",
                value=sample_query if sample_query else "Men stylish topwear under 1500",
                placeholder="e.g., Casual cotton shirts for men under ₹1500"
            )

        if st.button("Search", type="primary", use_container_width=True):
            if not query_input.strip():
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Finding matching products..."):
                    result = pipeline.query_text(
                        query=query_input,
                        manual_filters=manual_filters,
                        top_k=top_k,
                        use_auto_filter=auto_extract
                    )

                # Check if out of domain
                if result.get("guardrail_status") in ["out_of_domain", "flagged"]:
                    st.info(
                        "👋 **I am an apparel and fashion assistant.** "
                        "I can only help you explore clothing, styling recommendations, and products from our catalog. "
                        "Please search for shirts, pants, dresses, sarees, or style preferences!"
                    )
                else:
                    # Recommendation summary
                    if result.get("answer"):
                        header = "**Recommendation:**" if result.get("products") else "**Catalog Notice:**"
                        with st.container(border=True):
                            st.markdown(f"{header}\n\n{result['answer']}")

                    # Product Cards
                    products = result.get("products", [])
                    if products:
                        st.subheader(f"Matching Items ({len(products)})")
                        col1, col2 = st.columns(2)
                        for idx, p in enumerate(products):
                            with col1 if idx % 2 == 0 else col2:
                                render_product_card(p)

    # MODE 2: Visual Search (Image Upload)
    elif search_mode == "Visual Search (Image Match)":
        st.subheader("Visual Search")
        st.caption("Upload a clothing photo to find visually similar items in the catalog.")

        col_up, col_sample_img = st.columns(2)
        with col_up:
            uploaded_file = st.file_uploader("Upload an apparel image", type=["jpg", "jpeg", "png"])

        with col_sample_img:
            sample_files = ["15970.jpg", "10257.jpg", "10401.jpg", "10866.jpg", "11349.jpg"]
            chosen_sample = st.selectbox("Or select a catalog sample:", [""] + sample_files)

        img_to_search = None
        if uploaded_file:
            img_to_search = Image.open(uploaded_file)
            st.image(img_to_search, caption="Search Image", width=220)
        elif chosen_sample:
            sample_path = config.IMAGES_DIR / chosen_sample
            if sample_path.exists():
                img_to_search = Image.open(sample_path)
                st.image(img_to_search, caption=f"Sample ({chosen_sample})", width=220)

        if st.button("Find Matching Items", type="primary", disabled=img_to_search is None):
            with st.spinner("Finding visually similar apparel..."):
                result = pipeline.query_image(
                    image_input=img_to_search,
                    manual_filters=manual_filters,
                    top_k=top_k
                )

            if result.get("answer"):
                header = "**Visual Match Summary:**" if result.get("products") else "**Catalog Notice:**"
                with st.container(border=True):
                    st.markdown(f"{header}\n\n{result['answer']}")

            products = result.get("products", [])
            if products:
                st.subheader(f"Visually Similar Items ({len(products)})")
                col1, col2 = st.columns(2)
                for idx, p in enumerate(products):
                    with col1 if idx % 2 == 0 else col2:
                        render_product_card(p)



if __name__ == "__main__":
    main()
