# Week 8 Incident Response Plan

## 1. Purpose
Respond to possible silent model degradation, input drift, API failures, and unusual prediction behavior.

## 2. Monitoring signals
- API error rates and response latency.
- Prediction confidence and toxic probability.
- Input text statistics: character count, word count, mean word length,
  uppercase ratio, and punctuation ratio.
- Distribution drift alerts compared with the training-data reference baseline.
- Changes in incoming request volume.

## 3. Incident triggers
Investigate when:
- Drift monitoring repeatedly flags one or more input features.
- Prediction confidence changes substantially from the expected pattern.
- API error rates or latency rise unexpectedly.
- Labeled evaluation results show reduced precision, recall, or F1 score.
- Users report suspicious or incorrect predictions.

Drift alone is an investigation signal, not proof of accuracy loss.

## 4. Response procedure
1. Acknowledge the alert and record its time and request identifiers.
2. Inspect structured logs, drift statistics, input characteristics, and API health.
3. Determine whether the issue is data drift, malformed input, infrastructure failure,
   or a model-quality problem.
4. If predictions may be harmful or unreliable, notify the project owner and consider
   temporarily disabling predictions or routing requests for human review.
5. Evaluate a labeled validation dataset and compare precision, recall, F1 score,
   and confusion matrix with the approved baseline.
6. If needed, revert to the last known-good model and configuration.
7. Retrain or adjust the model only after validating representative data and reviewing
   the results.
8. Run regression tests, validation tests, and load tests before restoring service.
9. Document the root cause, impact, corrective action, and follow-up tasks.

## 5. Recovery and prevention
- Keep a versioned copy of the approved model and reference baseline.
- Protect logs and avoid recording unnecessary sensitive raw text.
- Review recurring drift alerts and thresholds.
- Evaluate the model periodically using labeled data.
- Never automatically retrain or replace a model based on drift alone.

## 6. Current project limitations
- The drift detector monitors text-feature distributions, not actual accuracy.
- The current test used synthetic requests and a small concurrent workload.
- Production thresholds must be calibrated against representative traffic.
- Request validation failures currently may not have a request ID.
