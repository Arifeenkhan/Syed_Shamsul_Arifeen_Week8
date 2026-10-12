"""Standard-library concurrent load test. Run after the Docker container starts."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

URL = "http://localhost:8000/predict"
WORKERS = 20
REQUESTS = 200
PAYLOADS = [
    {"text": "Thank you for your thoughtful contribution."},
    {"text": "You are terrible and nobody wants you here."},
    {"text": "I disagree with this opinion, but I respect your right to share it."},
]


def send_request(index: int) -> dict:
    encoded = json.dumps(PAYLOADS[index % len(PAYLOADS)]).encode("utf-8")
    request = Request(URL, data=encoded, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=15) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {"ok": response.status == 200, "status": response.status, "latency_ms": (time.perf_counter() - start) * 1000, "label": body.get("label")}
    except (HTTPError, URLError, TimeoutError) as error:
        return {"ok": False, "status": getattr(error, "code", "network_error"), "latency_ms": (time.perf_counter() - start) * 1000, "error": str(error)}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round((len(ordered) - 1) * p)))]


def main():
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(as_completed([executor.submit(send_request, i) for i in range(REQUESTS)]))
        results = [future.result() for future in results]
    elapsed = time.perf_counter() - started
    latencies = [result["latency_ms"] for result in results]
    summary = {
        "requests": REQUESTS, "concurrent_workers": WORKERS, "successful_requests": sum(r["ok"] for r in results),
        "failed_requests": sum(not r["ok"] for r in results), "elapsed_seconds": round(elapsed, 3),
        "requests_per_second": round(REQUESTS / elapsed, 2), "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(percentile(latencies, 0.95), 2), "max_latency_ms": round(max(latencies), 2),
    }
    Path("load_test/results.json").write_text(json.dumps({"summary": summary, "requests": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
