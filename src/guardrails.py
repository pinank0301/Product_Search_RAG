import re
import requests
from typing import Dict, Any, Tuple, List, Optional
from src import config

# Apparel Category Mappings
CATEGORY_KEYWORDS = {
    "Topwear": [
        "top", "tops", "topwear", "shirt", "shirts", "t-shirt", "t-shirts", "tshirt", "tshirts", 
        "tee", "tees", "polo", "polos", "hoodie", "hoodies", "sweatshirt", "sweatshirts", 
        "sweater", "sweaters", "jacket", "jackets", "kurta", "kurtas"
    ],
    "Bottomwear": [
        "bottom", "bottoms", "bottomwear", "pant", "pants", "trouser", "trousers", 
        "jean", "jeans", "shorts", "chinos", "trackpant", "trackpants", "jogger", "joggers", "skirt", "skirts"
    ],
    "Dress": [
        "dress", "dresses", "gown", "gowns", "maxi", "frock", "frocks", "one-piece", "jumpsuit", "jumpsuits"
    ],
    "Saree": [
        "saree", "sarees", "sari", "saris"
    ],
    "Apparel Set": [
        "set", "sets", "suit", "suits", "kurta set", "kurta sets", "apparel set", "apparel sets", "co-ord", "coord", "coords"
    ]
}

# Gender Keywords
GENDER_KEYWORDS = {
    "Men": ["men", "man", "male", "gents", "boy", "boys"],
    "Women": ["women", "woman", "female", "ladies", "girl", "girls"]
}

# Prompt Injection Patterns
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) (instructions|directions)",
    r"system prompt",
    r"reveal your (secret|instructions|system)",
    r"disregard the above",
    r"drop table",
    r"delete from",
]

def check_safety_guardrail(prompt: str) -> Tuple[bool, str]:
    """
    Check if the user prompt violates safety or injection guardrails.
    Returns (is_safe, reason).
    """
    # 1. Quick regex check for prompt injection
    lower_prompt = prompt.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower_prompt):
            return False, "Prompt rejected: detected potential prompt injection or unauthorized instruction."

    # 2. Llama Guard 3 8B check via Cloudflare Workers AI
    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{config.CLOUDFLARE_ACCOUNT_ID}/ai/run/{config.CLOUDFLARE_GUARD_MODEL}"
        headers = {
            "Authorization": f"Bearer {config.CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [{"role": "user", "content": prompt}]
        }
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        if res.ok:
            output = res.json().get("result", {}).get("response", "").strip()
            if "unsafe" in output.lower():
                return False, "Prompt flagged as unsafe by content safety policy."
    except Exception as e:
        # Fallback to permissive on network timeout for safety model, regex already applied
        print(f"Llama Guard check skipped due to error: {e}")

    return True, "Safe"


def extract_metadata_filters(query: str) -> Dict[str, Any]:
    """
    Extract structured metadata filters (Category, Gender, Price thresholds) from natural language query.
    Used for fast pre-filtering in NeonDB SQL before or instead of vector search.
    """
    lower = query.lower()
    filters: Dict[str, Any] = {}

    # 1. Extract Gender
    for gender, keywords in GENDER_KEYWORDS.items():
        pattern = r"\b(" + "|".join(keywords) + r")\b"
        if re.search(pattern, lower):
            filters["gender"] = gender
            break

    # 2. Extract Category
    for cat, keywords in CATEGORY_KEYWORDS.items():
        pattern = r"\b(" + "|".join(keywords) + r")\b"
        if re.search(pattern, lower):
            filters["category"] = cat
            break

    # 3. Extract Max Price
    # Patterns like: under 1000, below 1500, less than 2000, < 2000, under rs. 1500, under ₹1500, max 2000
    max_price_match = re.search(
        r"(?:under|below|less than|within|up to|max|<|<=|budget of?)\s*(?:rs\.?|inr|₹)?\s*(\d{3,6})",
        lower
    )
    if max_price_match:
        try:
            filters["max_price"] = int(max_price_match.group(1))
        except ValueError:
            pass

    # 4. Extract Min Price
    # Patterns like: above 1000, more than 1500, > 1000, min 2000
    min_price_match = re.search(
        r"(?:above|over|more than|greater than|>|>=|min(?:imum)?)\s*(?:rs\.?|inr|₹)?\s*(\d{3,6})",
        lower
    )
    if min_price_match:
        try:
            filters["min_price"] = int(min_price_match.group(1))
        except ValueError:
            pass

    # 5. Check if query is purely structured (e.g. "show me men topwear under 1000")
    # Clean words to see if there is substantial semantic intent left
    semantic_words = re.sub(
        r"\b(show|find|me|give|list|all|any|products?|items?|men|man|male|women|woman|female|topwear|bottomwear|dress|saree|suit|under|below|above|more|than|rs|inr|₹|\d+)\b",
        "",
        lower
    ).strip()
    filters["has_semantic_intent"] = len(semantic_words) > 3

    return filters


# Out of domain patterns for apparel shopping guardrail
OUT_OF_DOMAIN_PATTERNS = [
    (r"\b(?:code|coding|program|programming|python|javascript|java|c\+\+|html|css|sql|function|algorithm|script|regex|loop|class|method|def\s+\w+)\b", "programming or software coding"),
    (r"\b(?:calculate|derivative|integral|equation|math|algebra|geometry|physics|chemistry|formula|solve for)\b", "mathematics or scientific calculations"),
    (r"\b(?:recipe|ingredients|bake a|how to cook|culinary|food recipe)\b", "cooking recipes"),
    (r"\b(?:president|prime minister|capital of|who is the king|history of war|geography trivia)\b", "general trivia or politics"),
    (r"\b(?:medical diagnosis|symptoms of|cure for|prescribe medicine)\b", "medical advice"),
]

def check_domain_relevance(prompt: str) -> Tuple[bool, str]:
    """
    Ensure user queries stay within the apparel, fashion, shopping, and catalog domain.
    Rejects out-of-domain requests like programming, math, trivia, cooking, etc.
    """
    lower = prompt.lower().strip()
    
    # Check out-of-domain patterns
    for pattern, topic in OUT_OF_DOMAIN_PATTERNS:
        if re.search(pattern, lower):
            return False, (
                f"This request appears to be about {topic}. "
                "I am a personal Fashion & Apparel Shopping Assistant, so I can only help you explore clothing, "
                "style advice, outfit recommendations, and our product catalog. Please ask an apparel-related question!"
            )

    return True, "In domain"

# Unsupported catalog segments (catalog only carries Men and Women apparel)
UNSUPPORTED_SEGMENTS = {
    "Kids": [r"\b(kid|kids|child|children|toddler|baby|infant|boys?|girls?)\b"],
    "Footwear": [r"\b(shoe|shoes|sneaker|sneakers|footwear|heels|sandals|boots)\b"],
    "Accessories": [r"\b(watch|watches|handbag|handbags|sunglasses|jewelry|necklace|earrings)\b"],
    "Electronics": [r"\b(phone|laptop|headphone|headphones|charger|gadget)\b"]
}

def check_unsupported_segment(prompt: str) -> Optional[str]:
    """Check if the user is asking for a category/audience we don't carry (Kids, Shoes, Electronics)."""
    lower = prompt.lower()
    for segment, patterns in UNSUPPORTED_SEGMENTS.items():
        if any(re.search(p, lower) for p in patterns):
            return segment
    return None


def verify_grounding(response_text: str, retrieved_skus: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Verify that any SKU mentioned in the LLM response was actually retrieved.
    Prevents hallucinating nonexistent SKUs without crashing on variable-width regex look-behinds.
    """
    # 1. Look for explicit SKU tags like "SKU 15970", "SKU: 15970", "#15970"
    explicit_skus = re.findall(r"(?:sku|item|product)\s*[:#-]?\s*(\d{4,6})\b", response_text, re.IGNORECASE)
    if explicit_skus:
        cited_skus = explicit_skus
    else:
        # 2. Extract 4-6 digit tokens and filter out currency/prices safely without regex lookbehinds
        all_matches = re.finditer(r"\b(\d{4,6})\b", response_text)
        cited_skus = []
        for m in all_matches:
            start_idx = m.start()
            preceding = response_text[max(0, start_idx - 6) : start_idx].lower()
            if not any(curr in preceding for curr in ["₹", "$", "€", "£", "rs", "inr"]):
                cited_skus.append(m.group(1))

    if not cited_skus:
        return True, None

    valid_sku_set = set(retrieved_skus)
    hallucinated = [sku for sku in cited_skus if sku not in valid_sku_set]

    if hallucinated:
        return False, f"Grounding check warning: LLM mentioned SKU(s) {hallucinated} not found in retrieved context."
    return True, None

