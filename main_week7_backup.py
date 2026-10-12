"""FastAPI service for the Week 6 TF-IDF toxicity classifier."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import time

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "tfidf_logreg.joblib"


class PredictRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=10_000, description="Comment text to classify")]

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be blank")
        return cleaned


class PredictResponse(BaseModel):
    label: str
    toxic_probability: float
    threshold: float
    processing_time_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file is missing: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or {"vectorizer", "classifier"} - bundle.keys():
        raise RuntimeError("Model bundle must contain 'vectorizer' and 'classifier'.")
    app.state.vectorizer = bundle["vectorizer"]
    app.state.classifier = bundle["classifier"]
    yield


app = FastAPI(
    title="Toxic Comment Classification API",
    version="1.0.0",
    description="Week 7 Dockerized API for the Week 6 TF-IDF + Logistic Regression model.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request. Send JSON with a non-empty 'text' field of at most 10,000 characters.", "errors": exc.errors()},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "tfidf-logistic-regression"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    try:
        features = app.state.vectorizer.transform([payload.text])
        toxic_probability = float(app.state.classifier.predict_proba(features)[0, 1])
    except AttributeError as exc:
        raise HTTPException(status_code=503, detail="Model is not ready. Try again shortly.") from exc

    return PredictResponse(
        label="toxic" if toxic_probability >= 0.50 else "non-toxic",
        toxic_probability=round(toxic_probability, 6),
        threshold=0.50,
        processing_time_ms=round((time.perf_counter() - start) * 1000, 3),
    )
