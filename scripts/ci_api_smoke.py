"""Smoke-test a running API container (stdlib only; used by .github/workflows/ci.yml).

    python3 scripts/ci_api_smoke.py http://127.0.0.1:8000

Checks /health, then posts api/example_request.json (a real dev row) to /predict and compares the
score with the Python reference in reports/p4_web_stream.csv.gz (row 0, score_final) to 1e-9;
also checks that an invalid request is rejected with 422.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def call(url: str, body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main(base: str) -> int:
    for _ in range(30):  # wait for the container to start
        try:
            status, health = call(f"{base}/health")
            if status == 200:
                break
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(1)
    else:
        print("FAIL: /health did not answer")
        return 1
    print("health:", health)

    example = json.loads((REPO / "api" / "example_request.json").read_text())
    with gzip.open(REPO / "reports" / "p4_web_stream.csv.gz", "rt") as fh:
        expected = float(next(csv.DictReader(fh))["score_final"])
    status, pred = call(f"{base}/predict", example)
    print("predict:", status, {k: pred.get(k) for k in ("score", "decision", "threshold")})
    checks = {
        "status 200": status == 200,
        "score = Python reference (<= 1e-9)": abs(pred.get("score", -1) - expected) <= 1e-9,
        "decision follows the threshold": pred.get("decision")
        == ("step-up" if pred.get("score", 0) >= pred.get("threshold", 1) else "approve"),
        "three contributions": len(pred.get("top_contributions", [])) == 3,
        "invalid request -> 422": call(f"{base}/predict", {**example, "amount_usd": -1})[0] == 422,
    }
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"))
