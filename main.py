from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import json
import logging
import time
import uuid
from logging.handlers import RotatingFileHandler

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.monitoring import DriftMonitor, text_statistics

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "tfidf_logreg.joblib"
REFERENCE_PATH = BASE_DIR / "monitoring" / "reference_stats.json"
LOG_PATH = BASE_DIR / "logs" / "pipeline.jsonl"


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        entry.update(getattr(record, "event_data", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def create_logger():
    logger = logging.getLogger("week8.pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger


logger = create_logger()


def log_event(event, **fields):
    logger.info(event, extra={"event_data": fields})


class PredictRequest(BaseModel):
    text: Annotated[
        str, Field(min_length=1, max_length=10_000,
                   description="Comment text to classify")
    ]

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
    app.state.drift_monitor = None

    if REFERENCE_PATH.exists():
        with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
            reference_stats = json.load(f)
        app.state.drift_monitor = DriftMonitor(reference_stats)
        log_event(
            "drift_baseline_loaded",
            reference_records=len(reference_stats),
            reference_path=str(REFERENCE_PATH),
        )
    else:
        log_event("drift_baseline_missing", reference_path=str(REFERENCE_PATH))

    log_event("application_started", model_path=str(MODEL_PATH))
    yield
    log_event("application_stopped")


app = FastAPI(
    title="Toxic Comment Classification API",
    version="2.0.0",
    description="Week 8 API with structured logging and input drift monitoring.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log_event("request_validation_failed", path=request.url.path)
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request. Send JSON with a non-empty 'text' field of at most 10,000 characters.",
            "errors": jsonable_encoder(exc.errors()),
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "tfidf-logistic-regression",
        "drift_monitoring": (
            "active" if app.state.drift_monitor is not None else "unconfigured"
        ),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    request: Request,
    response: Response,
) -> PredictResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    log_event(
        "raw_input_received",
        request_id=request_id,
        text_length=len(payload.text),
    )

    try:
        stats = text_statistics(payload.text)
        features = app.state.vectorizer.transform([payload.text])
        log_event(
            "preprocessing_completed",
            request_id=request_id,
            input_statistics=stats,
        )

        toxic_probability = float(
            app.state.classifier.predict_proba(features)[0, 1]
        )
        label = "toxic" if toxic_probability >= 0.50 else "non-toxic"
        confidence = max(toxic_probability, 1.0 - toxic_probability)

        log_event(
            "inference_completed",
            request_id=request_id,
            label=label,
            toxic_probability=round(toxic_probability, 6),
            prediction_confidence=round(confidence, 6),
            threshold=0.50,
        )

        monitor = app.state.drift_monitor
        if monitor is not None:
            drift_result = monitor.observe(payload.text)
            log_event(
                "drift_monitoring_completed",
                request_id=request_id,
                **drift_result,
            )
        else:
            log_event(
                "drift_monitoring_completed",
                request_id=request_id,
                status="unconfigured_reference_missing",
                drift_detected=None,
                statistics=stats,
            )

        elapsed = round((time.perf_counter() - start) * 1000, 3)
        result = PredictResponse(
            label=label,
            toxic_probability=round(toxic_probability, 6),
            threshold=0.50,
            processing_time_ms=elapsed,
        )

        log_event(
            "response_sent",
            request_id=request_id,
            http_status=200,
            processing_time_ms=elapsed,
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        log_event(
            "prediction_failed",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500, detail="Prediction failed."
        ) from exc
