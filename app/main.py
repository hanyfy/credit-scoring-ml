"""API de scoring + petit front web. Lance : uvicorn app.main:app"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Union

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Credit scoring (German Credit)", version="0.1.0")


@lru_cache
def get_artifact() -> dict:
    if not Path(MODEL_PATH).exists():
        raise HTTPException(503, "Modèle absent : lancer `python -m scoring.train` d'abord.")
    return joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: Dict[str, Union[float, str]]


class PredictResponse(BaseModel):
    default_probability: float
    decision: str
    threshold: float
    risk_band: str


def validate(features: dict, schema: dict) -> pd.DataFrame:
    errors = []
    row = {}
    for c, meta in schema["numeric"].items():
        if c not in features:
            errors.append(f"champ manquant : {c}")
        else:
            try:
                row[c] = float(features[c])
            except (TypeError, ValueError):
                errors.append(f"{c} doit être numérique")
    for c, meta in schema["categorical"].items():
        if c not in features:
            errors.append(f"champ manquant : {c}")
        elif features[c] not in meta["values"]:
            errors.append(f"{c} : valeur inconnue {features[c]!r}")
        else:
            row[c] = features[c]
    if errors:
        raise HTTPException(422, errors)
    return pd.DataFrame([row])


def band(p: float) -> str:
    return "faible" if p < 0.2 else "modéré" if p < 0.4 else "élevé"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema(art: dict = Depends(get_artifact)):
    return art["schema"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, art: dict = Depends(get_artifact)):
    X = validate(req.features, art["schema"])
    p = float(art["model"].predict_proba(X)[0, 1])
    thr = art["threshold"]
    return PredictResponse(default_probability=round(p, 4), threshold=thr, risk_band=band(p),
                           decision="refus conseillé" if p >= thr else "accord conseillé")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
