# Week 8: Production AI Pipeline

## Project overview
This project extends the Week 7 toxic-comment classification API with structured
pipeline logging, input-distribution drift monitoring, validation error handling,
and concurrent-request testing.

## Architecture
- Model: TF-IDF and Logistic Regression.
- API: FastAPI.
- Input validation: Pydantic request schema.
- Logging: JSON Lines pipeline logs.
- Drift detection: Kolmogorov-Smirnov tests on text statistics.
- Baseline: Statistics computed from 5,000 training comments.
- Load test: 50 requests using 10 concurrent workers.

## Implemented features
1. Logs for raw input receipt, preprocessing, inference, drift monitoring, and response.
2. Structured logging for request-validation failures and startup.
3. Request identifiers for successful prediction pipeline events.
4. Reference text statistics loaded when the application starts.
5. Drift checks after collecting 30 recent requests.
6. API validation for blank, missing, and oversized text.
7. Health endpoint and prediction endpoint.
8. Incident-response plan for suspected silent degradation.

## Validation test results

| Test | Result |
|---|---|
| Valid comment | HTTP 200 |
| Blank comment | HTTP 422 |
| Missing text | HTTP 422 |
| Text longer than 10,000 characters | HTTP 422 |

## Concurrent-load test

| Metric | Result |
|---|---:|
| Requests | 50 |
| Concurrent workers | 10 |
| Successful responses | 50 |
| Failed responses | 0 |
| Total duration | 0.60 seconds |
| Throughput | 83.78 requests/second |
| Average latency | 112.22 ms |
| Median latency | 84.06 ms |
| 95th-percentile latency | 207.46 ms |

These are results from a small test in the Colab environment and are not a production
capacity guarantee.

## Drift-monitoring observations
The monitor collected a reference distribution from 5,000 training comments.
It began checking the recent-request window after 30 observations. In the test,
all five text-statistic features were flagged in several windows.

These alerts show that incoming text statistics differed from the reference sample.
They do not establish that model accuracy decreased. Accuracy degradation must be
checked with labeled evaluation data.

## Known limitations and future work
- Add request IDs to validation failures.
- Calibrate drift thresholds with representative production traffic.
- Add a labeled evaluation process for precision, recall, F1 score, and confusion matrix.
- Test longer sustained workloads and measure resource consumption.
- Secure logs and avoid retaining raw user comments unnecessarily.
- Add automated alerting and a tested rollback procedure.

## Conclusion
The Week 8 prototype implements pipeline observability, input validation,
statistical drift monitoring, and a basic concurrent-load test. Further evaluation
with representative labeled data and deployment-level testing is required before
production use.
