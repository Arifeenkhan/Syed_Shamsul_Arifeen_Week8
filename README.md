# Week 8 – Production AI Pipeline Monitoring

**Student Name:** Syed Shamsul Arifeen  
**Project:** Toxic Comment Classification API  
**Internship:** Avenrix Technology  
**Week:** 8

## 1. Project Overview

This project extends my Week 7 Toxic Comment Classification API into a production-style AI pipeline.

The system uses a TF-IDF vectorizer and Logistic Regression model to classify text comments. It is built with FastAPI and includes input validation, structured logging, input drift monitoring, and concurrent request testing.

## 2. Week 8 Objectives

- Extend the Week 7 API.
- Record pipeline events using structured JSON logs.
- Monitor prediction confidence and input statistics.
- Detect changes in incoming text compared with training data.
- Test API behavior under concurrent requests.
- Document an incident-response plan for possible model performance degradation.

## 3. Technologies Used

- Python
- FastAPI
- Scikit-learn
- TF-IDF
- Logistic Regression
- NumPy
- SciPy
- Docker
- Pytest

## 4. Main Features

### API Engineering
- REST API built with FastAPI.
- `/health` endpoint for health checks.
- `/predict` endpoint for predictions.
- Input validation for missing, blank, and oversized text.
- Structured error responses.

### Pipeline Logging
The pipeline records important events, including:

- Raw input received
- Preprocessing completed
- Inference completed
- Drift monitoring completed
- Response sent
- Validation or prediction errors

### Input Drift Monitoring
The system compares incoming text statistics with reference statistics generated from training comments.

It monitors:
- Character count
- Word count
- Mean word length
- Uppercase ratio
- Punctuation ratio

**Important:** Drift indicates a change in input data. It does not automatically prove that model accuracy has decreased.

### Concurrent Load Testing
The API was tested with 50 requests and 10 concurrent workers.

## 5. Testing Results

| Test | Result |
|---|---|
| Valid prediction request | Passed — HTTP 200 |
| Blank text validation | Passed — HTTP 422 |
| Missing text validation | Passed — HTTP 422 |
| Oversized text validation | Passed — HTTP 422 |
| Concurrent requests | 50/50 successful |
| Average response latency | 112.22 ms |
| 95th-percentile latency | 207.46 ms |
| Measured throughput | 83.78 requests/second |

These results were measured in a Google Colab test environment. They are not a guarantee of production performance.

## 6. Project Structure

The repository should contain the following important files and folders:

```text
Syed_Shamsul_Arifeen_Week8/
├── app/
│   ├── main.py
│   ├── main_week7_backup.py
│   └── monitoring.py
├── model/
│   └── tfidf_logreg.joblib
├── monitoring/
│   └── reference_stats.json
├── logs/
│   └── pipeline.jsonl
├── tests/
│   └── test_api.py
├── load_test/
│   └── concurrent_test.py
├── Dockerfile
├── requirements.txt
├── INCIDENT_RESPONSE.md
├── WEEK8_REPORT.md
└── README.md
```

## 7. Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the API

Run this command from the project root directory:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Open API Documentation

Visit:

http://127.0.0.1:8000/docs

### Check API Health

Visit:

http://127.0.0.1:8000/health

### Build the Docker Image

```bash
docker build -t week8-toxic-api .
```

### Run the Docker Container

```bash
docker run -p 8000:8000 week8-toxic-api
```

## 8. Incident Response

If drift alerts increase or model performance is suspected to have degraded:

1. Review pipeline logs and drift-monitoring results.
2. Check for unusual input patterns and API errors.
3. Evaluate the model using newly labeled data.
4. Compare current performance with the baseline.
5. If necessary, roll back to a known-good model.
6. Document the incident and corrective actions.

See [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) for the incident-response plan.

## 9. Project Report

See [WEEK8_REPORT.md](WEEK8_REPORT.md) for the implementation details, monitoring approach, and testing results.

## 10. Conclusion

This project demonstrates how an AI model API can be extended with production-oriented monitoring, structured logging, input drift detection, validation, Docker support, and concurrent load testing.

**Author:** Syed Shamsul Arifeen  
**Repository:** Syed_Shamsul_Arifeen_Week8
