import os
import json
import requests
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from src import config
from src.embeddings import CloudflareTextEmbeddings, LocalCLIPEmbeddings
from src import db
from src.guardrails import (
    check_safety_guardrail, 
    check_domain_relevance, 
    check_unsupported_segment, 
    extract_metadata_filters, 
    verify_grounding
)

class CloudflareLLM:
    """Wrapper for Cloudflare Workers AI Llama-3.1-8B-Instruct model."""

    def __init__(
        self,
        account_id: str = config.CLOUDFLARE_ACCOUNT_ID,
        api_token: str = config.CLOUDFLARE_API_TOKEN,
        model_name: str = config.CLOUDFLARE_LLM_MODEL,
    ):
        self.account_id = account_id
        self.api_token = api_token
        self.model_name = model_name
        self.endpoint = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_name}"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Call Cloudflare LLM chat completion endpoint."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        if not data.get("success", False):
            raise RuntimeError(f"Cloudflare LLM Error: {data.get('errors')}")
        return data.get("result", {}).get("response", "").strip()


class ProductRAGPipeline:
    """Unified Multimodal RAG Pipeline for Apparel Product Search and Recommendations."""

    def __init__(self):
        self.text_embedder = CloudflareTextEmbeddings()
        self.clip_embedder = LocalCLIPEmbeddings()
        self.llm = CloudflareLLM()

    def query_text(
        self,
        query: str,
        manual_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        use_auto_filter: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a natural language user query with guardrails, hybrid metadata pre-filtering,
        pgvector similarity search, and grounded LLM generation.
        """
        # Step 1: Input Safety Guardrail
        is_safe, reason = check_safety_guardrail(query)
        if not is_safe:
            return {
                "answer": f"⚠️ Request Blocked by Safety Guardrail: {reason}",
                "products": [],
                "filters_applied": {},
                "retrieval_mode": "Blocked (Safety Policy)",
                "guardrail_status": "flagged",
            }

        # Step 2: Domain Relevance Guardrail
        is_in_domain, domain_reason = check_domain_relevance(query)
        if not is_in_domain:
            return {
                "answer": f"⚠️ Out-of-Domain Request:\n\n{domain_reason}",
                "products": [],
                "filters_applied": {},
                "retrieval_mode": "Blocked (Out of Domain)",
                "guardrail_status": "out_of_domain",
            }

        # Step 3: Extract metadata filters from query
        extracted_filters = extract_metadata_filters(query) if use_auto_filter else {}
        
        # Merge manual filters (from UI) with auto-extracted filters
        combined_filters = {}
        if manual_filters:
            combined_filters.update({k: v for k, v in manual_filters.items() if v})
        for k, v in extracted_filters.items():
            if k not in combined_filters and v is not None:
                combined_filters[k] = v

        # Step 3: Retrieval Strategy Decision
        # If the user query is purely a metadata lookup without semantic adjectives (e.g. "show men topwear under 1500"),
        # we can fulfill it via direct SQL query without calling vector embeddings, saving time and compute!
        has_semantic_intent = extracted_filters.get("has_semantic_intent", True)
        clean_filters = {k: v for k, v in combined_filters.items() if k in ["category", "gender", "max_price", "min_price"]}

        # Check for unsupported segment in query (e.g. "Kids", "Footwear", "Accessories", "Electronics")
        unsupported_segment = check_unsupported_segment(query)
        if unsupported_segment:
            return {
                "answer": (
                    f"We're sorry, but we currently do not carry {unsupported_segment.lower()} apparel in our catalog. "
                    "PROSEARCH specializes exclusively in Men's and Women's apparel (Topwear, Bottomwear, Dresses, Sarees, and Apparel Sets)."
                ),
                "products": [],
                "filters_applied": clean_filters,
                "retrieval_mode": "No Match (Unsupported Department)",
                "guardrail_status": "passed",
            }

        if not has_semantic_intent and clean_filters:
            retrieval_mode = "Direct Metadata Filter (Fast SQL)"
            products = db.search_metadata_only(clean_filters, limit=top_k)
        else:
            retrieval_mode = "Hybrid (Metadata Pre-filter + pgvector Cosine Search)"
            query_vector = self.text_embedder.embed_query(query)
            products = db.search_by_text_embedding(query_vector, filters=clean_filters, limit=top_k)

        # If no products match the query in the database, gracefully inform the user without showing other options
        if not products:
            reasons = []
            if clean_filters.get("category"):
                reasons.append(f"in {clean_filters['category']}")
            if clean_filters.get("gender"):
                reasons.append(f"for {clean_filters['gender']}")
            if clean_filters.get("max_price"):
                reasons.append(f"under ₹{clean_filters['max_price']:,}")

            constraint_desc = f" ({', '.join(reasons)})" if reasons else ""
            return {
                "answer": (
                    f"We're sorry, but no products matching your search '{query}'{constraint_desc} are currently available in our catalog. "
                    "Please try adjusting your budget, filters, or search terms."
                ),
                "products": [],
                "filters_applied": clean_filters,
                "retrieval_mode": "No Match",
                "guardrail_status": "passed",
            }

        # Step 4: Format Context for LLM Augmentation
        context_items = []
        for idx, p in enumerate(products, 1):
            tag = p.get("tagline", "")
            desc = p.get("description", "")
            notes = f"{tag} - {desc}".strip(" -")
            context_items.append(
                f"{idx}. SKU: {p['sku']} | Title: {p['title']} | Category: {p['category']} ({p['gender']}) | Price: INR {p['price_inr']} | Style Notes: {notes}"
            )
        context_str = "\n".join(context_items)

        # Step 5: LLM Generation with Strict Grounding
        budget_clause = f"The user specified a maximum budget of ₹{clean_filters['max_price']:,}." if clean_filters.get("max_price") else ""
        availability_instructions = (
            f"BUDGET ACCURACY: All {len(products)} products listed below are strictly within the user's budget. Never claim or imply any item is above budget. {budget_clause}"
        )

        system_prompt = (
            "You are the PROSEARCH Personal Fashion Stylist. Provide expert, stylish shopping recommendations based strictly on the retrieved products.\n\n"
            f"{availability_instructions}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            f"1. You MUST review and introduce ALL {len(products)} products listed in the retrieved context. Do NOT omit or skip any product.\n"
            "2. For each product, present:\n"
            "   - SKU (e.g., SKU 1855)\n"
            "   - Title and Category\n"
            "   - Price in INR (e.g., ₹410)\n"
            "   - 1-2 sentences on styling tips and why it matches or serves as a great alternative.\n"
            "3. NEVER mention technical metrics like 'similarity score', 'cosine distance', or 'vectors'.\n"
            "4. Keep the tone courteous, helpful, and warm, formatted with clean bullet points."
        )

        user_prompt = (
            f"User Search: \"{query}\"\n\n"
            f"Retrieved Catalog Products ({len(products)} items):\n"
            f"{context_str}\n\n"
            f"Present your personalized fashion recommendations covering all {len(products)} items:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_response = self.llm.generate(messages, max_tokens=900)

        # Step 6: Output Grounding Guardrail
        retrieved_skus = [p["sku"] for p in products]
        is_grounded, grounding_warning = verify_grounding(llm_response, retrieved_skus)
        if not is_grounded and grounding_warning:
            llm_response += f"\n\n*[Notice: {grounding_warning}]*"

        return {
            "answer": llm_response,
            "products": products,
            "filters_applied": clean_filters,
            "retrieval_mode": retrieval_mode,
            "guardrail_status": "passed",
        }

    def query_image(
        self,
        image_input: Union[str, Image.Image],
        manual_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Visual search: given an image file or PIL Image, compute CLIP embedding
        and find visually similar apparel items in NeonDB pgvector.
        Automatically classifies garment category (and gender) to focus visual search.
        """
        # Encode image to 512-dim vector
        image_vector = self.clip_embedder.embed_image(image_input)

        clean_filters = {k: v for k, v in (manual_filters or {}).items() if v}

        # Auto-detect garment category and gender from the photo
        detected_cat, cat_conf, detected_gender, gender_conf = self.clip_embedder.classify_image(image_input)

        # Apply category focus if not manually overridden by user
        auto_filters = dict(clean_filters)
        applied_auto_cat = False
        applied_auto_gender = False

        if "category" not in auto_filters and cat_conf >= 0.35:
            auto_filters["category"] = detected_cat
            applied_auto_cat = True

        if "gender" not in auto_filters and gender_conf >= 0.65:
            auto_filters["gender"] = detected_gender
            applied_auto_gender = True

        # First try searching with auto-detected filters
        products = db.search_by_image_embedding(image_vector, filters=auto_filters, limit=top_k)

        # If too few results with gender constraint, relax gender first
        if len(products) < min(2, top_k) and applied_auto_gender:
            fallback_filters = {k: v for k, v in auto_filters.items() if k != "gender"}
            fallback_products = db.search_by_image_embedding(image_vector, filters=fallback_filters, limit=top_k)
            if len(fallback_products) > len(products):
                products = fallback_products
                auto_filters = fallback_filters

        if not products:
            return {
                "answer": f"We're sorry, but no visually similar products matching this {detected_cat.lower()} item are currently available in our catalog.",
                "products": [],
                "filters_applied": auto_filters,
                "retrieval_mode": "Visual Search (CLIP)",
                "guardrail_status": "passed",
            }

        # Context summary for LLM
        context_items = []
        for p in products:
            context_items.append(f"- SKU: {p['sku']} | {p['title']} ({p['category']}, {p['gender']}) at ₹{p['price_inr']} - {p.get('tagline', '')}")
        context_str = "\n".join(context_items)

        system_prompt = (
            "You are a personal fashion stylist for PROSEARCH. The user uploaded a photo looking for visually similar clothes.\n"
            "Review the visually matching products retrieved from our catalog and present a friendly, concise recommendation.\n"
            "For each product, mention its SKU, title, category, and price in INR, highlighting its visual style and fit.\n"
            "Never mention technical metrics like 'similarity score', 'cosine distance', or 'vectors'."
        )

        detected_notes = []
        if applied_auto_cat:
            detected_notes.append(f"detected garment: {detected_cat}")
        if applied_auto_gender:
            detected_notes.append(f"style: {detected_gender}'s apparel")
        meta_note = f" (Focus: {', '.join(detected_notes)})" if detected_notes else ""

        user_prompt = (
            f"The visual similarity search matched the following items{meta_note}:\n"
            f"{context_str}\n\n"
            "Provide a friendly, stylish recommendation summary of these matching items:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_response = self.llm.generate(messages, max_tokens=500)

        return {
            "answer": llm_response,
            "products": products,
            "filters_applied": auto_filters,
            "detected_category": detected_cat,
            "category_confidence": cat_conf,
            "detected_gender": detected_gender,
            "retrieval_mode": "Visual Search (CLIP ViT-B/32)",
            "guardrail_status": "passed",
        }

    def query_cross_modal(
        self,
        query: str,
        manual_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Cross-modal search: text description matched directly against image embeddings in CLIP space.
        """
        # Input safety
        is_safe, reason = check_safety_guardrail(query)
        if not is_safe:
            return {
                "answer": f"⚠️ Request Blocked by Guardrail: {reason}",
                "products": [],
                "retrieval_mode": "Blocked (Safety Policy)",
                "guardrail_status": "flagged",
            }

        # Domain relevance
        is_in_domain, domain_reason = check_domain_relevance(query)
        if not is_in_domain:
            return {
                "answer": f"⚠️ Out-of-Domain Request:\n\n{domain_reason}",
                "products": [],
                "retrieval_mode": "Blocked (Out of Domain)",
                "guardrail_status": "out_of_domain",
            }

        # Encode query using CLIP text encoder
        clip_text_vector = self.clip_embedder.embed_text_query(query)
        clean_filters = {k: v for k, v in (manual_filters or {}).items() if v}
        products = db.search_by_image_embedding(clip_text_vector, filters=clean_filters, limit=top_k)

        return {
            "answer": f"Cross-modal visual search results matching description: '{query}'",
            "products": products,
            "filters_applied": clean_filters,
            "retrieval_mode": "Cross-Modal (CLIP Text-to-Image)",
            "guardrail_status": "passed",
        }

