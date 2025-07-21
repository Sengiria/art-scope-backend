import torch
import numpy as np
import math
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import open_clip
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

# Load model + preprocessing
device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device).eval()

# Lazy global cache
data_cache = None

def load_data():
    global data_cache
    if data_cache is None:
        print("Loading embeddings and labels...")
        data = np.load("data/art_data.npz", allow_pickle=True)

        embeddings = data["embeddings"].astype(np.float32)
        labels_df = pd.DataFrame({
            "filename": data["filenames"],
            "artist_name": data["artist_names"],
            "genre": data["genres"]
        })

        data_cache = (embeddings, labels_df)
    return data_cache

# Inference function
def embed_uploaded_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_input)

        # Check for zero vector before normalization
        norm = features.norm(dim=-1, keepdim=True)
        if torch.any(norm == 0) or torch.any(torch.isnan(norm)):
            raise ValueError("Embedding resulted in zero or NaN norm.")

        features /= norm

    output = features.cpu().squeeze().numpy()

    # Final sanity check
    if np.isnan(output).any() or np.all(output == 0):
        raise ValueError("Final embedding is NaN or all-zero.")

    return output

def find_top_matches(uploaded_vector: np.ndarray, top_k: int = 5) -> List[dict]:
    embeddings, labels_df = load_data()
    sims = cosine_similarity([uploaded_vector], embeddings)[0]

    sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
    top_indices = sims.argsort()[-top_k:][::-1]

    results = []
    for i in top_indices:
        score = float(sims[i])
        if math.isnan(score) or math.isinf(score):
            score = 0.0

        result = {
            "filename": os.path.basename(labels_df.iloc[i]["filename"]),
            "artist_name": str(labels_df.iloc[i].get("artist_name", "unknown")),
            "genre": str(labels_df.iloc[i].get("genre", "Unknown")),
            "score": round(score, 4),
        }
        results.append(result)

    return {"result": results[0]}