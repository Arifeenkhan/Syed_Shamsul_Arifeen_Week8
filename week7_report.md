# Week 7: Dockerized toxicity-classification API report

**Student:** SYED SHAMSUL ARIFEEN

## API implementation and validation

The service exposes a FastAPI `POST /predict` endpoint. It receives JSON in the form `{"text": "comment text"}` and returns a toxicity label, probability, threshold, and processing time. The service loads the Week 6 TF-IDF + Logistic Regression model once at startup rather than retraining for each request.

Input validation rejects a missing `text` field, blank text, and text longer than 10,000 characters. Invalid input receives HTTP 422 with a clear error message. `GET /health` returns HTTP 200 and confirms that the service is running.

Example request:

```json
{"text": "Thank you for your thoughtful contribution."}
```

## Dockerization

The service runs in a `python:3.11-slim` image. The Dockerfile installs pinned dependencies, copies the API and model bundle, exposes port 8000, and runs as a non-root user. It starts with Uvicorn using `0.0.0.0` so that the API is accessible from the host machine.

Build and run commands:

```bash
docker build -t week7-toxic-api .
docker run --rm -p 8000:8000 week7-toxic-api
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Concurrent load test

The included test sends 200 requests using 20 concurrent worker threads to the Dockerized endpoint. Run it only after the container is running:

```bash
python load_test/concurrent_test.py
```

Paste the printed values after your run:

| Measure | Result |
|---|---:|
| Total requests | 200 |
| Concurrent workers | 20 |
| Successful requests | 200 |
| Failed requests | 0 |
| Throughput (requests/second) | 114.45 |
| Median latency (ms) | 166.64 |
| 95th-percentile latency (ms) | 202.65 |
| Maximum latency (ms) | 283.23 |

## Interpretation

The Week 6 model had a median single-comment inference time of 2.65 ms and an 8.73 MB model size. Under 20 concurrent workers, the Dockerized API completed all 200 requests successfully in 1.747 seconds, with no failures. Throughput was 114.45 requests per second. Median end-to-end latency was 166.64 ms, the 95th-percentile latency was 202.65 ms, and the maximum latency was 283.23 ms. These values are higher than the Week 6 single-model measurement because they include HTTP request parsing, JSON serialization, concurrent queueing, and container overhead. The zero failure count demonstrates that the service remained reliable during this test.

The TF-IDF model is appropriate for this API because its small size and low CPU inference cost support inexpensive deployment. The API includes validation and a health endpoint so malformed requests do not cause server errors and operational checks can verify service availability.
