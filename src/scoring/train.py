"""Entraîne, compare et évalue deux modèles de scoring ; écrit modèle, métriques et graphiques."""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # noqa: E402

from scoring.data import CATEGORICAL, FEATURES, NUMERIC, load  # noqa: E402
from scoring.monitoring import psi  # noqa: E402

SEED = 42
COST_FN, COST_FP = 5, 1  # matrice de coûts fournie avec le jeu : accorder un mauvais crédit coûte 5x un refus à tort


def build_models():
    lr_pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    gb_pre = ColumnTransformer([("num", "passthrough", NUMERIC),
                                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                                 CATEGORICAL)])
    return {
        "logistic_regression": Pipeline([("pre", lr_pre), ("clf", LogisticRegression(C=0.3, max_iter=2000))]),
        "gradient_boosting": Pipeline([("pre", gb_pre), ("clf", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED,
            categorical_features=list(range(len(NUMERIC), len(FEATURES)))))]),
    }


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_cost(y, p, thr) -> float:
    pred = p >= thr
    fn = np.sum((~pred) & (y == 1))  # mauvais payeur accepté
    fp = np.sum(pred & (y == 0))     # bon payeur refusé
    return float(COST_FN * fn + COST_FP * fp) / len(y)


def best_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmin([expected_cost(y, p, t) for t in grid]))])


def main(out_dir: str = ".") -> dict:
    out = Path(out_dir)
    (out / "models").mkdir(exist_ok=True)
    (out / "docs").mkdir(exist_ok=True)
    X, y = load(str(out / "data/openml"))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, model in build_models().items():
        oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        thr = best_threshold(ytr.values, oof)  # seuil choisi SANS regarder le jeu de test
        auc = roc_auc_score(yte, p)
        results[name] = {
            "cv_auc_train": round(float(roc_auc_score(ytr, oof)), 3),
            "test_auc": round(float(auc), 3), "test_gini": round(float(2 * auc - 1), 3),
            "test_ks": round(ks_stat(yte.values, p), 3), "test_brier": round(float(brier_score_loss(yte, p)), 3),
            "threshold": round(thr, 2), "test_cost_per_applicant": round(expected_cost(yte.values, p, thr), 3),
        }
        fitted[name] = (model, p)

    results["baselines_test_cost_per_applicant"] = {
        "approve_everyone": round(expected_cost(yte.values, np.zeros(len(yte)), 2.0), 3),
        "reject_everyone": round(expected_cost(yte.values, np.ones(len(yte)), 0.0), 3),
    }
    chosen = max(("logistic_regression", "gradient_boosting"), key=lambda n: results[n]["cv_auc_train"])
    results["chosen_model"] = chosen  # choisi sur la validation croisée du train, pas sur le test
    model, p_test = fitted[chosen]
    results["psi_train_vs_test_scores"] = round(psi(model.predict_proba(Xtr)[:, 1], p_test), 3)
    results["n_train"], results["n_test"], results["bad_rate"] = len(Xtr), len(Xte), round(float(y.mean()), 3)

    # --- graphiques ---
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for n, (_, p) in fitted.items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, label=f"{n} (AUC {results[n]['test_auc']})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbe ROC (jeu de test)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "docs/roc.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p_test[yte.values == 0], bins=20, alpha=.6, label="bons payeurs")
    ax.hist(p_test[yte.values == 1], bins=20, alpha=.6, label="mauvais payeurs")
    ax.axvline(results[chosen]["threshold"], color="k", ls="--", label="seuil de décision")
    ax.set(xlabel="Probabilité de défaut prédite", ylabel="Dossiers", title=f"Distribution des scores ({chosen})")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "docs/score_distribution.png", dpi=130); plt.close(fig)

    imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=20, random_state=SEED)
    order = np.argsort(imp.importances_mean)[-10:]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh([FEATURES[i] for i in order], imp.importances_mean[order], xerr=imp.importances_std[order])
    ax.set(xlabel="Baisse d'AUC en permutant la variable", title="Importance par permutation (jeu de test)")
    fig.tight_layout(); fig.savefig(out / "docs/feature_importance.png", dpi=130); plt.close(fig)
    results["top_features"] = [FEATURES[i] for i in order[::-1][:5]]

    # --- artefacts ---
    schema = {"numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
                          for c in NUMERIC},
              "categorical": {c: {"values": sorted(X[c].unique().tolist()), "mode": X[c].mode()[0]}
                              for c in CATEGORICAL}}
    model.fit(X, y)  # refit sur toutes les données pour le service ; les métriques ci-dessus viennent du split
    joblib.dump({"model": model, "schema": schema, "threshold": results[chosen]["threshold"]},
                out / "models/model.joblib")
    (out / "docs/metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
