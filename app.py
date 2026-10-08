from __future__ import annotations
import asyncio, io, json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pipeline import ExtremeHandwritingPipeline, MAX_IMAGE_PIXELS, RecognitionError

ROOT = Path(__file__).parent
app = FastAPI(title="Extreme Handwriting Digitizer", version="1.0.0")
pipeline = ExtremeHandwritingPipeline()
app.mount("/data", StaticFiles(directory=ROOT / "data"), name="data")

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "extreme-digitizer",
        "version": app.version,
        "recognition": pipeline.engine_status,
    }

@app.get("/api/samples")
def samples():
    items = []
    for p in sorted((ROOT / "data").glob("*.png")):
        meta = p.with_suffix(".json")
        items.append(json.loads(meta.read_text()) if meta.exists() else {"id": p.stem, "name": p.name, "url": f"/data/{p.name}"})
    return {"samples": items}

@app.post("/api/digitize")
async def digitize(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload a PNG, JPG, or other image file.")
    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image is larger than 12 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise HTTPException(400, f"Could not read image: {e}")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(413, "Image dimensions exceed the 40-megapixel processing limit.")
    try:
        result = await asyncio.to_thread(pipeline.process, image)
    except RecognitionError as e:
        raise HTTPException(502, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return JSONResponse(result)

@app.post("/api/feedback")
async def feedback(block_id: str = Form(...), corrected_text: str = Form(...), original_text: str = Form("")):
    if not corrected_text.strip():
        raise HTTPException(400, "Correction cannot be empty.")
    path = ROOT / "feedback" / "corrections.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"block_id": block_id, "original_text": original_text, "corrected_text": corrected_text.strip()}, ensure_ascii=False) + "\n")
    return {"status": "saved", "message": "Correction added to the review buffer."}
