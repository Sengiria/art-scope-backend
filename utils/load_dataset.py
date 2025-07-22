import os
import pandas as pd

images_folder = os.path.join("data", "resized")

image_data = []
for file in os.listdir(images_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        artist_name = "_".join(file.split("_")[:-1]) 
        image_path = os.path.join(images_folder, file)
        image_data.append({
            "filename": image_path.replace("\\", "/"),
            "artist_name": artist_name
        })

print(f"Collected {len(image_data)} images")

# Convert to DataFrame
metadata_df = pd.DataFrame(image_data)

# Save
os.makedirs("data", exist_ok=True)
metadata_df.to_csv("data/metadata.csv", index=False)

print(f"Saved metadata for {len(metadata_df)} images.")
