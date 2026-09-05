import os
import sys
import torch

print("=" * 60)
print("Downloading & caching CLIP ViT-B/32 model weights...")
print("=" * 60)

try:
    # pyrefly: ignore [missing-import]
    import clip
    # Set single thread to avoid memory spike during build
    torch.set_num_threads(1)
    
    # Pre-download and cache model in ~/.cache/clip/
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    print("SUCCESS: CLIP ViT-B/32 model is downloaded and cached successfully!")
except Exception as e:
    print(f"WARNING: Could not pre-download CLIP model during build: {e}")
    sys.exit(0)  # Don't fail the build if network glitch occurs
