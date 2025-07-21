import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import open_clip

# Paths
METADATA_PATH = "data/metadata.csv"
EMBEDDINGS_PATH = "data/embeddings.npy"
LABELS_PATH = "data/labels.csv"

# Load metadata
metadata = pd.read_csv(METADATA_PATH)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
model = model.to(device).eval()

# Prepare image transform
def get_embedding(img_path: str):
    try:
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().squeeze().numpy()
    except Exception as e:
        print(f"Failed to process: {img_path}\n{e}")
        return None

# Run embedding
all_embeddings = []
valid_rows = []

print("Generating image embeddings...")
for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    emb = get_embedding(row["filename"])
    if emb is not None:
        all_embeddings.append(emb)
        valid_rows.append(row)

# Convert to arrays and save
print(f"Embedded {len(all_embeddings)} images.")

np.save(EMBEDDINGS_PATH, np.array(all_embeddings))
pd.DataFrame(valid_rows).to_csv(LABELS_PATH, index=False)

print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
print(f"Saved labels to: {LABELS_PATH}")
