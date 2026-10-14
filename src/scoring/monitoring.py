"""Population Stability Index (PSI) : stabilité d'une distribution entre deux échantillons."""
import numpy as np


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI entre `expected` (référence) et `actual`. Bornes de classes = quantiles de la référence.
    Repères usuels (conventions, pas des lois) : < 0,1 stable ; 0,1-0,25 à surveiller ; > 0,25 dérive."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
