from pydantic import BaseModel, Field
from typing import Dict, List


class PredictionItem(BaseModel):
    genre: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    genre: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    top_predictions: List[PredictionItem]
    segment_count: int = Field(ge=1)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: List[str]


class ModelInfoResponse(BaseModel):
    model_name: str
    num_classes: int
    classes: List[str]
    sample_rate: int
    n_mels: int
    n_fft: int
    hop_length: int
    segment_duration: float
