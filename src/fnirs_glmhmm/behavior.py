"""VTC smoothing and in/out-of-the-zone labelling."""

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, windows


def smooth_vtc(vtc: np.ndarray | pd.Series, length: int = 20) -> np.ndarray:
    """Gaussian smoothing of the variance time course, following Esterman et al.

    Zero-phase filter so peaks don't shift relative to the trial they came from.
    """
    vtc = np.asarray(vtc, dtype=float)
    w = windows.gaussian(length, std=length / 2) / 2
    return filtfilt(w, np.sum(w), vtc)


def zone_labels(
    vtc_smoothed: np.ndarray, threshold: float | None = None
) -> tuple[np.ndarray, float]:
    """Binary zone label per trial: 1 = out of the zone, 0 = in the zone.

    Defaults to a median split within the run, which is the usual convention.
    Returns the labels plus whatever threshold was used, so it can be reused
    across runs of the same subject.
    """
    if threshold is None:
        threshold = float(np.median(vtc_smoothed))
    return (vtc_smoothed > threshold).astype(int), threshold


def zone_series(
    events: pd.DataFrame,
    length: int = 20,
    threshold: float | None = None,
    vtc_col: str = "VTC",
) -> pd.DataFrame:
    """Add smoothed VTC and zone label columns to an event table."""
    out = events.copy()
    out["VTC_smoothed"] = smooth_vtc(out[vtc_col], length=length)
    out["out_of_zone"], _ = zone_labels(out["VTC_smoothed"].to_numpy(), threshold)
    return out
