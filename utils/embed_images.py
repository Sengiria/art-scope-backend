import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import open_clip

METADATA_PATH = "data/metadata.csv"
EMBEDDINGS_PATH = "data/embeddings.npy"
LABELS_PATH = "data/labels.csv"
metadata = pd.read_csv(METADATA_PATH)

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
model = model.to(device).eval()

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

all_embeddings = []
valid_rows = []

print("Generating image embeddings...")
for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    emb = get_embedding(row["filename"])
    if emb is not None:
        all_embeddings.append(emb)
        valid_rows.append(row)

valid_df = pd.DataFrame(valid_rows, columns=metadata.columns)

print(f"Embedded {len(all_embeddings)} images.")
np.save(EMBEDDINGS_PATH, np.array(all_embeddings))
valid_df.to_csv(LABELS_PATH, index=False)

print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
print(f"Saved labels to: {LABELS_PATH}")
print("valid_rows sample:", valid_df.head(3).to_dict(orient='records'))

np.savez_compressed("data/art_data.npz",
    embeddings=np.array(all_embeddings, dtype=np.float32),
    filenames=valid_df["filename"].values,
    artist_names=valid_df["artist_name"].values,
)
print("Saved compressed bundle to data/art_data.npz")