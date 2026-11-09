"""FastAPI service: score one transaction with the project's logistic model.

    uv run uvicorn api.app:app --port 8000        # from the repo (model files in reports/)
    docker run -p 8000:8000 fraud-console-api     # see Dockerfile

Model files are read from $MODEL_DIR (default: <repo>/reports): p4_web_model.json (the same exported
pipeline the website runs) and p2_threshold.json (the step-up threshold).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import Contribution, Health, Prediction, Transaction
from api.scoring import Model, load_model

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "reports"


@lru_cache(maxsize=1)
def get_model() -> Model:
    return load_model(Path(os.environ.get("MODEL_DIR", DEFAULT_DIR)))


app = FastAPI(
    title="Fraud console API",
    version="1.0.0",
    description=(
        "Scores a card transaction with the cost-aware logistic model from "
        "https://github.com/Natpaphonnn/fraud-console. Trained on a public CC0 dataset of "
        "unspecified provenance; the threshold comes from illustrative assumed costs "
        "(not bank data). "
        "A portfolio project, not a production fraud system."
    ),
)


@app.get("/health", response_model=Health)
def health() -> Health:
    m = get_model()
    return Health(status="ok", model=m.spec_id, n_features=len(m.features), threshold=m.threshold)


@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction) -> Prediction:
    m = get_model()
    row = txn.model_dump()
    unknown = {f: row[f] for f, lut in m.categories.items() if row[f] not in lut}
    if unknown:
        allowed = {f: sorted(m.categories[f]) for f in unknown}
        raise HTTPException(422, detail={"unknown_categories": unknown, "allowed": allowed})
    contribs = m.contributions(row)
    logit = m.intercept + sum(c for _, c in contribs)
    score = m.score(row)
    top = sorted(contribs, key=lambda fc: abs(fc[1]), reverse=True)[:3]
    return Prediction(
        score=score,
        decision="step-up" if score >= m.threshold else "approve",
        threshold=m.threshold,
        threshold_basis=m.threshold_basis,
        logit=logit,
        top_contributions=[Contribution(feature=f, value=row[f], contribution=c) for f, c in top],
        model=m.spec_id,
    )
