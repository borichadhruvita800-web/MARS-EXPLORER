import numpy as np

HAZARD_LABELS = ["Low (0-5°)", "Moderate (5-15°)", "High (15-30°)", "Very high (30°+)"]


def classify_hazard(slope):
    """0 low, 1 moderate, 2 high, 3 very high. NaN stays NaN."""
    hazard = np.full(slope.shape, np.nan)
    hazard[(slope >= 0) & (slope < 5)] = 0
    hazard[(slope >= 5) & (slope < 15)] = 1
    hazard[(slope >= 15) & (slope < 30)] = 2
    hazard[slope >= 30] = 3
    return hazard


def hazard_stats(hazard):
    valid = np.count_nonzero(np.isfinite(hazard))
    return {label: (np.count_nonzero(hazard == i) / valid * 100 if valid else 0.0)
            for i, label in enumerate(HAZARD_LABELS)}