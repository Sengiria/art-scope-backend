from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from .inference import embed_uploaded_image, find_top_matches
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
import asyncio

app = FastAPI(default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/ping")
def ping():
    return {"message": "pong"}

async def process_image(contents: bytes) -> Image.Image:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: Image.open(io.BytesIO(contents)).convert("RGB"))

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = await process_image(contents)
        vector = embed_uploaded_image(image)
        results = find_top_matches(vector, top_k=5)
        return {"results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
