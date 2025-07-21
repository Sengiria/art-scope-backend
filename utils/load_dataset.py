import kagglehub
import os
import pandas as pd
import zipfile

# Download latest version
path = kagglehub.dataset_download("ikarus777/best-artworks-of-all-time")
artist_csv_path = os.path.join(path, "artists.csv")

artists_df = pd.read_csv(artist_csv_path)

# Filter out bad entries
artists_df = artists_df.dropna(subset=["genre"])
artists_df = artists_df[artists_df["paintings"] > 1]
artists_df = artists_df.drop_duplicates(subset=["name"])

images_zip_path = os.path.join(path, "images.zip")
images_folder = os.path.join(path, "images")

# Unzip images if not already done
if not os.path.exists(images_folder):
    with zipfile.ZipFile(images_zip_path, "r") as zip_ref:
        zip_ref.extractall(images_folder)
print("Images extracted to:", images_folder)

# Fix nested structure if exists
subdirs = os.listdir(images_folder)
if "images" in subdirs:
    images_folder = os.path.join(images_folder, "images")

print("Found artist folders:", os.listdir(images_folder))

# Walk the image folders
image_data = []
for artist_dir in os.listdir(images_folder):
    artist_path = os.path.join(images_folder, artist_dir)
    if not os.path.isdir(artist_path):
        continue

    for file in os.listdir(artist_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(artist_path, file)
            image_data.append({
                "filename": image_path,
                "artist_name": artist_dir
            })

print(f"Collected {len(image_data)} images")

# Convert to DataFrame
metadata_df = pd.DataFrame(image_data)

# Merge with genre
metadata_df = metadata_df.merge(
    artists_df[["name", "genre"]],
    left_on="artist_name",
    right_on="name",
    how="left"
)

metadata_df = metadata_df.drop(columns=["name"])

# Save
os.makedirs("data", exist_ok=True)
metadata_df.to_csv("data/metadata.csv", index=False)

print(f"Saved metadata for {len(metadata_df)} images.")
