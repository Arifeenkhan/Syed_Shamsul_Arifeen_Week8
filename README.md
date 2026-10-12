# Week 7 - Dockerized Toxic Comment API

**Student:** SYED SHAMSUL ARIFEEN

This project packages the Week 6 TF-IDF + Logistic Regression toxicity classifier as a FastAPI web service.

## Files

- `app/main.py`: API implementation with `/health` and `/predict`.
- `model/tfidf_logreg.joblib`: trained Week 6 model bundle.
- `tests/test_api.py`: API and validation tests.
- `Dockerfile`: reproducible container definition.
- `load_test/concurrent_test.py`: 200-request, 20-worker concurrent load test.
- `report/week7_report.md`: implementation and performance report.

## Run with Docker

```bash
docker build -t week7-toxic-api .
docker run --rm -p 8000:8000 week7-toxic-api
```

Open `http://localhost:8000/docs` in a browser and test the API.

Example request:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"You are very helpful."}'
```

## Test and load test

With the container running, open a second terminal in this folder:

```bash
python load_test/concurrent_test.py
```

The test prints a summary and saves full results to `load_test/results.json`. Copy the summary values into `report/week7_report.md`.

## GitHub submission

Create a public repository named `Syed_Shamsul_Arifeen_Week7`, upload this entire project folder, then submit the repository URL as required.
