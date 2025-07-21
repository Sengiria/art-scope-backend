# ArtScope Backend – Artist & Style Identifier API

This is the backend API for **ArtScope**, a machine learning-powered project that identifies the **artist** and potential **style** of a given painting using image embeddings and cosine similarity and a dataset of famous 50 painters.

## Features

- Upload a painting image
- Get top matching artists and artworks based on visual similarity
- Powered by OpenCLIP (`ViT-B-32`, `laion2b_s34b_b79k`)
- Based on Kaggle's *Best Artworks of All Time* dataset
- FastAPI + Uvicorn API server

## How It Works

1. We embed each artwork in the dataset using OpenCLIP.
2. On image upload, the API:
   - Preprocesses the image
   - Embeds it using the same OpenCLIP model
   - Finds the closest artworks via cosine similarity
3. It returns the top N results with metadata (artist, score, etc.)

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/Sengiria/art-scope.git
cd art-scope/backend
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
import kagglehub
kagglehub.download("ikarus777/best-artworks-of-all-time")
```

The dataset will be cached under ~/.cache/kagglehub/...

### 5. Generate embeddings (first time only)

```bash
python embed_dataset.py
```

This will create:

- data/embeddings.npy – image embeddings
- data/labels.csv – metadata about each image

### 6. Run the backend

```bash
uvicorn main:app --reload
```

Then open:
http://localhost:8000/docs for Swagger UI to test image upload and get predictions.

## Project Structure

<pre>
backend/
├── data/
│   ├── embeddings.npy
│   └── labels.csv
├── main.py        
├── embed_dataset.py
├── model.py
└── requirements.txt
</pre>

## Sample Usage

Upload a .jpg or .png file using the /predict endpoint in Swagger UI. You'll get the first result of a ranked list of similar artworks:

```bash
 {
    "artist_name": "Vincent_van_Gogh",
    "score": 0.9419,
    "genre": "nan",
    "filename": "Vincent_van_Gogh_368.jpg"
},
```

## Credits
- OpenCLIP
- Best Artworks of All Time Dataset