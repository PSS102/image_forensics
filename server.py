"""
FastAPI REST API Server
------------------------
Exposes the forensic pipeline as an HTTP API so it can be consumed
by the React dashboard or any external client.

Endpoints:
  POST /analyze        — upload an image and run full forensic analysis
  GET  /report/{id}    — download the generated PDF report
  GET  /results/{id}   — get JSON results for an analysis
  GET  /health         — health check
"""

import os
import uuid
import json
import base64
import shutil
import tempfile
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import pipeline
import sys
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import ForensicPipeline

app = FastAPI(
    title="Image Forensics API",
    description="Detects image manipulation using multi-module forensic analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory result store (use Redis/DB in production)
analysis_store: dict = {}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = ForensicPipeline(output_dir=OUTPUT_DIR)


def encode_image(img_array: np.ndarray) -> Optional[str]:
    """Encode numpy image array to base64 PNG string."""
    if img_array is None:
        return None
    try:
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        success, buffer = cv2.imencode(".png", img_array)
        if success:
            return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except Exception:
        pass
    return None


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image and run full forensic analysis.

    Returns a JSON payload with:
      - analysis_id
      - overall_score
      - verdict
      - manipulation_type
      - module_scores
      - base64-encoded visualization images
      - report_url (to download PDF)
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, TIFF, BMP, WEBP"
        )

    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    analysis_id = str(uuid.uuid4())

    try:
        # Run analysis
        results = pipeline.analyze(tmp_path, generate_report=True)

        # Encode visualization images as base64
        encoded_images = {}
        for label, img_array in results.get("image_outputs", {}).items():
            encoded = encode_image(img_array)
            if encoded:
                encoded_images[label] = encoded

        # Build response
        response = {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "timestamp": results["timestamp"],
            "image_size": results["image_size"],
            "image_format": results["image_format"],
            "overall_score": results["overall_score"],
            "verdict": results["verdict"],
            "manipulation_type": results["manipulation_type"],
            "summary": results["summary_text"],
            "module_scores": {
                name: {
                    "score": info["score"],
                    "key_finding": info["key_finding"],
                    "flags": info["flags"],
                    "findings": info["findings"],
                }
                for name, info in results["module_scores"].items()
            },
            "visualizations": encoded_images,
            "report_url": f"/report/{analysis_id}",
            "metadata_flags": results.get("module_scores", {})
                .get("Metadata Analysis", {}).get("flags", []),
        }

        # Store for later retrieval
        analysis_store[analysis_id] = {
            "response": response,
            "report_path": results.get("report_path"),
            "tmp_path": tmp_path,
        }

        return JSONResponse(content=response)

    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/report/{analysis_id}")
async def download_report(analysis_id: str):
    """Download the PDF forensic report for a given analysis."""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")

    report_path = analysis_store[analysis_id].get("report_path")
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not yet generated")

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=os.path.basename(report_path),
    )


@app.get("/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Retrieve JSON results for a completed analysis."""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return JSONResponse(content=analysis_store[analysis_id]["response"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
