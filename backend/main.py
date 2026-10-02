import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

try:
    from .audio_service import audio_service
    from .explainability_service import explainability_service
    from .model_service import model_service
    from .schemas import HealthResponse, ModelInfoResponse, PredictionItem, PredictionResponse
except ImportError:
    from audio_service import audio_service
    from explainability_service import explainability_service
    from model_service import model_service
    from schemas import HealthResponse, ModelInfoResponse, PredictionItem, PredictionResponse

app = FastAPI(
    title="GENGROGRAM API",
    description="Music genre classification using Mel-spectrograms and CustomCNN.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGIN",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Predicted-Genre", "X-Confidence", "X-Segment-Index", "X-Onset-Seconds"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def validate_filename(filename: str | None):
    if not filename:
        raise HTTPException(status_code=400, detail="No filename was provided.")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Use WAV, MP3, FLAC, OGG, or M4A.")


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=model_service.model is not None, classes=model_service.class_names)


@app.get("/api/model/info", response_model=ModelInfoResponse)
def model_info():
    processor = model_service.audio_processor
    return ModelInfoResponse(
        model_name="CustomCNN",
        num_classes=len(model_service.class_names),
        classes=model_service.class_names,
        sample_rate=processor.sample_rate,
        n_mels=processor.n_mels,
        n_fft=processor.n_fft,
        hop_length=processor.hop_length,
        segment_duration=processor.segment_duration,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    validate_filename(file.filename)
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        temp_path = audio_service.save_upload(file_bytes, file.filename)
        try:
            result = model_service.predict_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        sorted_predictions = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
        return PredictionResponse(
            genre=result["genre"],
            confidence=float(result["confidence"]),
            probabilities={g: float(p) for g, p in result["probabilities"].items()},
            top_predictions=[PredictionItem(genre=g, probability=float(p)) for g, p in sorted_predictions[:3]],
            segment_count=result["segment_count"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@app.post("/api/spectrogram")
async def spectrogram(file: UploadFile = File(...)):
    validate_filename(file.filename)
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        temp_path = audio_service.save_upload(file_bytes, file.filename)
        try:
            image_bytes, onset_seconds = audio_service.create_spectrogram_image(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        return Response(content=image_bytes, media_type="image/png", headers={"X-Onset-Seconds": f"{onset_seconds:.4f}"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Spectrogram generation failed: {str(exc)}")


@app.post("/api/explainability")
async def explainability(file: UploadFile = File(...), target_genre: str | None = Form(default=None)):
    validate_filename(file.filename)
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        temp_path = audio_service.save_upload(file_bytes, file.filename)
        try:
            result = explainability_service.generate_gradcam(temp_path, target_genre=target_genre)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        return Response(
            content=result["image"],
            media_type="image/png",
            headers={
                "X-Predicted-Genre": result["genre"],
                "X-Confidence": str(result["confidence"]),
                "X-Segment-Index": str(result["segment_index"]),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(exc)}")


@app.get("/api/evaluation/summary")
def evaluation_summary():
    evaluation_file = PROJECT_ROOT / "models" / "evaluation_results.json"
    if not evaluation_file.exists():
        raise HTTPException(status_code=404, detail="Evaluation results were not found.")
    with open(evaluation_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/evaluation/confusion-matrix")
def confusion_matrix():
    image_file = PROJECT_ROOT / "results" / "confusion_matrix.png"
    if not image_file.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix image was not found.")
    return Response(content=image_file.read_bytes(), media_type="image/png")
