"""Chargement du jeu public German Credit (UCI Statlog, 1 000 dossiers) via OpenML."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

# Variables liées au sexe / à la nationalité : volontairement exclues du modèle.
EXCLUDED = ["personal_status", "foreign_worker"]
NUMERIC = ["duration", "credit_amount", "installment_commitment", "residence_since",
           "age", "existing_credits", "num_dependents"]
CATEGORICAL = ["checking_status", "credit_history", "purpose", "savings_status", "employment",
               "other_parties", "property_magnitude", "other_payment_plans", "housing", "job",
               "own_telephone"]
FEATURES = NUMERIC + CATEGORICAL


def load(data_home: str = "data/openml"):
    """Retourne (X, y) avec y = 1 pour un mauvais payeur ('bad'), 0 sinon."""
    Path(data_home).mkdir(parents=True, exist_ok=True)
    frame = fetch_openml("credit-g", version=1, as_frame=True, data_home=data_home).frame
    y = (frame["class"] == "bad").astype(int)
    X = frame[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    return X, y
