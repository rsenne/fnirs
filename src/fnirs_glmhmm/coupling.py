"""Trial-level coupling between a subject's neural HMM states and their behaviour.

The GLM-HMM is fit per subject, so everything here is within-subject first: one
slope per subject per outcome, and the group claim is a test across those slopes.
Nothing is pooled at the trial level, which also means no state ever has to mean
the same thing for two different people.

Behaviour never enters the fit, so every association scored here is out of sample.
Trials are matched to the neural sample `lag_s` seconds after onset, which is where
that trial's haemodynamic response lands; the lag is a free parameter and worth
profiling rather than guessing once.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from fnirs_glmhmm import behavior, io
from fnirs_glmhmm.neural import trial_samples
from fnirs_glmhmm.zones import fit_zone_hmm

# outcome -> the trials it is defined on
OUTCOMES = {
    "commission": "nogo",
    "omission": "go",
    "log_rt": "hit",
    "vtc": "hit",
}


def coupling_frame(runs, posteriors, lag_s: float = 4.0) -> pd.DataFrame:
    """One row per trial inside the analysis window, carrying the state posteriors."""
    frames = []
    for run, post in zip(runs, posteriors, strict=True):
        idx = trial_samples(run, lag_s)
        keep = idx >= 0
        ev = run.events.loc[keep].reset_index(drop=True)
        rc = ev["response_code"].to_numpy()
        df = pd.DataFrame(
            {
                "sub": run.sub,
                "run": run.run,
                "trial": np.flatnonzero(keep),
                "onset": ev["onset"].to_numpy(float),
                "sample": idx[keep],
                "vtc": ev["VTC"].to_numpy(float),
                "nogo": (ev["trial_type"] == "mnt").to_numpy(),
                "commission": (rc == -2).astype(float),
                "omission": (rc == -1).astype(float),
                "hit": rc == 1,
            }
        )
        df["go"] = ~df["nogo"]
        # reaction_time is 0, not NaN, on no-press trials.
        rt = ev["reaction_time"].to_numpy(float)
        df["log_rt"] = np.log(rt, out=np.full(len(rt), np.nan), where=rt > 0)
        probs = post[idx[keep]]
        for k in range(probs.shape[1]):
            df[f"p{k}"] = probs[:, k]
        df["state"] = probs.argmax(axis=1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def subset(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Trials the outcome is defined on: no-go for commissions, go for omissions,
    hits for RT and VTC. Hits only for VTC keeps interpolated no-press values out."""
    return df[df[OUTCOMES[outcome]]].dropna(subset=[outcome])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope. On a binary outcome this is a linear probability model, which
    holds up better than logistic when a subject has only a handful of errors."""
    if len(x) < 10 or x.std() < 1e-9:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _contrast(x: np.ndarray, y: np.ndarray) -> float:
    """Outcome in the upper half of the predictor minus the lower half.

    Same shape as the in/out-of-the-zone contrast the VTC literature reports, and
    it does not care whether the relationship is linear.
    """
    hi = x > np.median(x)
    if hi.sum() < 5 or (~hi).sum() < 5:
        return np.nan
    return float(y[hi].mean() - y[~hi].mean())


def subject_coupling(df: pd.DataFrame, predictors, outcomes=tuple(OUTCOMES)) -> pd.DataFrame:
    """One row per (subject, outcome, predictor)."""
    rows = []
    for outcome in outcomes:
        d = subset(df, outcome)
        for sub, g in d.groupby("sub"):
            y = g[outcome].to_numpy(float)
            for p in predictors:
                if p not in g:
                    continue
                x = g[p].to_numpy(float)
                rows.append(
                    dict(
                        sub=sub,
                        outcome=outcome,
                        predictor=p,
                        slope=_slope(x, y),
                        contrast=_contrast(x, y),
                        n=len(g),
                    )
                )
    return pd.DataFrame(rows)


def group_test(
    rows: pd.DataFrame, by=("outcome", "predictor"), value: str = "slope"
) -> pd.DataFrame:
    """Second level: are the per-subject effects consistently signed?

    Wilcoxon signed-rank over subjects, which is what the VTC analyses in
    findings.md use, so the two are directly comparable.
    """
    out = []
    for keys, g in rows.groupby(list(by)):
        v = g[value].dropna().to_numpy()
        rec = dict(zip(by, keys if isinstance(keys, tuple) else (keys,), strict=True))
        rec.update(
            n_subjects=len(v),
            mean=float(v.mean()) if len(v) else np.nan,
            median=float(np.median(v)) if len(v) else np.nan,
            n_positive=int((v > 0).sum()),
            p=float(wilcoxon(v).pvalue) if len(v) >= 6 else np.nan,
            value=value,
        )
        out.append(rec)
    return pd.DataFrame(out)


def zone_reference(sub: str, num_states: int = 2, smooth_length: int = 20) -> pd.DataFrame:
    """Per-trial VTC zone measures for one subject, to score the neural states against.

    `p_out_zone` is the zone HMM of zones.py (raw VTC, k = 2, highest-VTC state);
    `out_of_zone_split` is the classic smoothed median split.
    """
    events = io.load_all_events(sub)
    vtc = [e["VTC"].to_numpy() for e in events]
    fit = fit_zone_hmm(vtc, num_states=num_states)
    frames = []
    for run_idx, (v, p) in enumerate(zip(vtc, fit.out_of_zone_prob(), strict=True), start=1):
        split, _ = behavior.zone_labels(behavior.smooth_vtc(v, length=smooth_length))
        frames.append(
            pd.DataFrame(
                {
                    "sub": sub,
                    "run": run_idx,
                    "trial": np.arange(len(v)),
                    "p_out_zone": p,
                    "out_of_zone_split": split.astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
