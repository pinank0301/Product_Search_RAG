import time
import requests
import torch
from typing import List, Union, Tuple
from PIL import Image
# pyrefly: ignore [missing-import]
from langchain_core.embeddings import Embeddings
from src import config

class CloudflareTextEmbeddings(Embeddings):
    """LangChain Embeddings implementation for Cloudflare Workers AI Text Embeddings."""

    def __init__(
        self,
        account_id: str = config.CLOUDFLARE_ACCOUNT_ID,
        api_token: str = config.CLOUDFLARE_API_TOKEN,
        model_name: str = config.CLOUDFLARE_TEXT_EMBED_MODEL,
        batch_size: int = 20,
        max_retries: int = 3,
    ):
        self.account_id = account_id
        self.api_token = api_token
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.endpoint = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_name}"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json={"text": texts},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                if not data.get("success", False):
                    raise RuntimeError(f"Cloudflare API error: {data.get('errors')}")
                result_data = data.get("result", {}).get("data", [])
                if not result_data:
                    raise ValueError(f"No embedding data returned in response: {data}")
                return result_data
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Failed to fetch embeddings from Cloudflare after {self.max_retries} attempts: {e}")
                time.sleep(1.5 * attempt)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of product document texts in batches."""
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = self._call_api(batch)
            all_embeddings.extend(batch_vectors)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        result = self._call_api([text])
        return result[0]


class LocalCLIPEmbeddings:
    """CLIP Model wrapper for generating 512-dim multimodal embeddings for apparel images & text."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        self.model_name = "ViT-B/32"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._preprocess = None

    def _load_model(self):
        if self._model is None:
            # pyrefly: ignore [missing-import]
            import clip
            import torch
            # Limit CPU threads to prevent memory explosion on cloud hosting
            if self.device == "cpu":
                try:
                    torch.set_num_threads(1)
                except Exception:
                    pass
            self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
            self._model.eval()
        return self._model, self._preprocess

    def embed_image(self, image_input: Union[str, Image.Image]) -> List[float]:
        """Generate 512-dim normalized vector for an image file path or PIL Image."""
        import torch
        import gc
        model, preprocess = self._load_model()
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        # Downscale large user images to prevent high RAM consumption
        if image.width > 512 or image.height > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)

        image_tensor = preprocess(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            result = image_features.cpu().squeeze().tolist()

        del image_tensor, image_features
        gc.collect()
        return result

    def embed_images_batch(self, image_inputs: List[Union[str, Image.Image]], batch_size: int = 16) -> List[List[float]]:
        """Batch encode images."""
        import torch
        import gc
        model, preprocess = self._load_model()
        all_features = []
        for i in range(0, len(image_inputs), batch_size):
            batch_inputs = image_inputs[i : i + batch_size]
            tensors = []
            for img in batch_inputs:
                if isinstance(img, str):
                    pil_img = Image.open(img).convert("RGB")
                else:
                    pil_img = img.convert("RGB")
                if pil_img.width > 512 or pil_img.height > 512:
                    pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                tensors.append(preprocess(pil_img))
            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.inference_mode():
                features = model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            all_features.extend(features.cpu().tolist())
            del batch_tensor, features
            gc.collect()
        return all_features

    def embed_text_query(self, text: str) -> List[float]:
        """Encode text query into CLIP joint embedding space for cross-modal text-to-image matching."""
        # pyrefly: ignore [missing-import]
        import clip
        import torch
        model, _ = self._load_model()
        text_tokens = clip.tokenize([text], truncate=True).to(self.device)
        with torch.inference_mode():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().squeeze().tolist()

    def classify_image(
        self,
        image_input: Union[str, Image.Image]
    ) -> Tuple[str, float, str, float]:
        """
        Zero-shot classification to detect garment category and gender from an apparel photo.
        Returns (category, category_confidence, gender, gender_confidence).
        """
        # pyrefly: ignore [missing-import]
        import clip
        import torch
        import gc
        model, preprocess = self._load_model()
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        if image.width > 512 or image.height > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)

        image_tensor = preprocess(image).unsqueeze(0).to(self.device)

        categories = ["Topwear", "Bottomwear", "Dress", "Saree", "Apparel Set"]
        cat_prompts = [
            "a photo of topwear, a t-shirt, shirt, polo, blouse, or top",
            "a photo of bottomwear, pants, trousers, jeans, sweatpants, or shorts",
            "a photo of a dress, gown, or frock",
            "a photo of a traditional Indian saree",
            "a photo of an apparel set, tracksuit, or matching suit"
        ]

        genders = ["Men", "Women"]
        gender_prompts = [
            "a photo of men clothing, men apparel, or a man",
            "a photo of women clothing, women apparel, or a woman"
        ]

        with torch.inference_mode():
            cat_tokens = clip.tokenize(cat_prompts).to(self.device)
            cat_logits, _ = model(image_tensor, cat_tokens)
            cat_probs = cat_logits.softmax(dim=-1).squeeze().tolist()

            gender_tokens = clip.tokenize(gender_prompts).to(self.device)
            gender_logits, _ = model(image_tensor, gender_tokens)
            gender_probs = gender_logits.softmax(dim=-1).squeeze().tolist()

        best_cat_idx = int(torch.tensor(cat_probs).argmax())
        best_gender_idx = int(torch.tensor(gender_probs).argmax())

        del image_tensor, cat_tokens, gender_tokens
        gc.collect()

        return (
            categories[best_cat_idx],
            float(cat_probs[best_cat_idx]),
            genders[best_gender_idx],
            float(gender_probs[best_gender_idx])
        )



