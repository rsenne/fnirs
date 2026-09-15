"""VTC smoothing, the signed VTC, and in/out-of-the-zone labelling."""

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


def signed_vtc(events: pd.DataFrame, rt_col: str = "reaction_time") -> np.ndarray:
    """VTC before the absolute value. Negative is fast, positive is slow.

    Standardisation includes RT = 0 non-responses in the mean and standard
    deviation. Non-response trials then take the most recent responded trial's
    deviation; leading non-responses take the first responded trial's value.

    Compare `abs(signed_vtc(e))` with `e["VTC"]` to check the reconstruction.
    Fit the result with `zones.fit_zone_hmm(..., transform="none")`.
    """
    rt = events[rt_col].to_numpy(dtype=float)
    responded = rt > 0
    if not responded.any():
        raise ValueError("no responded trials, nothing to centre on")

    z = (rt - rt.mean()) / rt.std(ddof=0)
    idx = np.arange(len(rt))
    src = np.maximum.accumulate(np.where(responded, idx, -1))
    src[src < 0] = idx[responded][0]  # use the first response for leading non-responses
    return z[src]


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
