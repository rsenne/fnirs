"""Network-level fNIRS features and task design for the neural GLM-HMM.

Everything between the cedalion parcel pickles and the (emissions, inputs) pair
dynamax wants: probe-coverage masking, aggregation to Yeo networks, filtering and
downsampling, the HRF-convolved task design, and nuisance regression.

The model is fit to the neural timeseries with the task as input, so behaviour
never enters the features. Event tables ride along on `RunData` only because the
validation needs to map trials back onto neural samples.
"""

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import gamma

from fnirs_glmhmm import io

# Schaefer 17-network prefixes to Yeo-7. TempPar has no Yeo-7 home and is kept as
# its own feature rather than folded into Default, so the DMN feature that the
# main hypothesis rests on stays clean.
YEO7 = {
    "VisCent": "Vis",
    "VisPeri": "Vis",
    "SomMotA": "SomMot",
    "SomMotB": "SomMot",
    "DorsAttnA": "DorsAttn",
    "DorsAttnB": "DorsAttn",
    "SalVentAttnA": "SalVentAttn",
    "SalVentAttnB": "SalVentAttn",
    "LimbicA": "Limbic",
    "LimbicB": "Limbic",
    "ContA": "Cont",
    "ContB": "Cont",
    "ContC": "Cont",
    "DefaultA": "Default",
    "DefaultB": "Default",
    "DefaultC": "Default",
    "TempPar": "TempPar",
}

# Limbic is in the parcellation but not in the data: the median subject has one
# covered parcel there and eight subjects have none, so it never becomes a feature.
# Seven networks x two chromophores gives D = 14.
FEATURE_NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Cont", "Default", "TempPar")
CHROMOPHORES = ("HbO", "HbR")

COVERAGE_SD = 1e-7  # HbO sd below this means the parcel is outside probe sensitivity
BAND = (0.01, 0.2)  # keeps the ~0.12 Hz no-go response, leaves Mayer waves in
TARGET_FS = 1.0
DRIFT_CUTOFF = 0.01  # DCT drift basis, on top of the highpass

# Task-locked analysis window, measured from the first trial onset of the run.
# 449 trials at 0.8 s is 359 s, but neural runs are not all long enough for that:
# 355 s keeps 60 of the 62 runs (both dropped runs are sub-751).
WINDOW_S = 355.0

INPUT_NAMES = ("commission", "correct_rejection", "rt_mod")


@dataclass
class RunData:
    """One run's emissions and inputs, plus what the validation needs to align trials."""

    sub: str
    run: int
    y: np.ndarray  # (T, D) z-scored network features
    x: np.ndarray  # (T, M) z-scored task regressors
    t0: float  # run-clock time of sample 0, i.e. the first trial onset
    target_fs: float
    n_covered: int
    feature_names: list[str]
    input_names: list[str]
    events: pd.DataFrame = field(repr=False)

    @property
    def n_samples(self) -> int:
        return self.y.shape[0]


def _values(da) -> np.ndarray:
    """Strip pint units; cedalion hands back Quantity-backed DataArrays."""
    d = da.data
    return np.asarray(getattr(d, "magnitude", d), dtype=float)


def network_labels(parcels, temppar: str = "separate") -> np.ndarray:
    """Yeo-7 network per parcel. `temppar="default"` folds TempPar into Default."""
    out = []
    for p in parcels:
        net = YEO7[str(p).split("_")[0]]
        out.append("Default" if (net == "TempPar" and temppar == "default") else net)
    return np.array(out)


def covered_mask(runs, thresh: float = COVERAGE_SD) -> np.ndarray:
    """Parcels inside probe sensitivity, judged by HbO sd in *every* run.

    Parcels the probe cannot see sit at ~1e-15 rather than at zero, so this is a
    gap of several orders of magnitude and the threshold is not delicate.
    """
    sd = np.min([_values(r.sel(chromo="HbO")).std(axis=-1) for r in runs], axis=0)
    return sd > thresh


def hrf_kernel(fs: float, length_s: float = 32.0) -> np.ndarray:
    """SPM-style canonical double gamma, peak-normalised.

    The same kernel goes on both chromophores; HbR picks up its sign from the
    fitted weights rather than from a flipped regressor.
    """
    t = np.arange(0, length_s, 1.0 / fs)
    h = gamma.pdf(t, 6.0) - gamma.pdf(t, 16.0) / 6.0
    return h / np.abs(h).max()


def convolve_hrf(sticks: np.ndarray, fs: float, length_s: float = 32.0) -> np.ndarray:
    h = hrf_kernel(fs, length_s)
    return np.convolve(sticks, h)[: len(sticks)]


def _sticks(onsets, t: np.ndarray, amplitudes=None) -> np.ndarray:
    """Impulse train on the neural timebase, one sample per event."""
    s = np.zeros(len(t))
    onsets = np.asarray(onsets, dtype=float)
    amplitudes = np.ones(len(onsets)) if amplitudes is None else np.asarray(amplitudes, float)
    idx = np.searchsorted(t, onsets)
    keep = idx < len(t)
    np.add.at(s, idx[keep], amplitudes[keep])
    return s


def design_matrix(events: pd.DataFrame, t: np.ndarray, fs: float) -> np.ndarray:
    """HRF-convolved task regressors on the native neural timebase, (n_samples, 3).

    The plan's `nogo` regressor is deliberately absent. Every mountain trial is
    either a commission error or a correct rejection, so nogo is exactly the sum
    of the two columns here and the four-column design is singular. Recover the
    no-go main effect afterwards as a weighted sum of the two.

    `rt_mod` is go trials only, amplitude-modulated by log RT z-scored within the
    run. Omissions carry no RT (the column is 0, not NaN) so they are left out.
    """
    mnt = (events["trial_type"] == "mnt").to_numpy()
    rc = events["response_code"].to_numpy()
    onset = events["onset"].to_numpy(dtype=float)

    hit = ~mnt & (rc == 1)
    log_rt = np.log(events["reaction_time"].to_numpy(dtype=float)[hit])
    amp = (log_rt - log_rt.mean()) / log_rt.std()

    cols = [
        _sticks(onset[mnt & (rc == -2)], t),
        _sticks(onset[mnt & (rc == 0)], t),
        _sticks(onset[hit], t, amp),
    ]
    return np.column_stack([convolve_hrf(c, fs) for c in cols])


def bandpass(x: np.ndarray, fs: float, band=BAND, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth along the last axis."""
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def resample_window(
    x: np.ndarray, t: np.ndarray, t0: float, window_s: float, target_fs: float
) -> np.ndarray:
    """Crop to `window_s` from `t0` and land on a uniform grid at `target_fs`.

    Plain interpolation is enough: the signal is already lowpassed at 0.2 Hz, well
    under the 0.5 Hz Nyquist of the 1 Hz grid, so there is nothing left to alias.
    Cropping happens after filtering so the filter's edge transients stay outside
    the analysis window.
    """
    grid = t0 + np.arange(round(window_s * target_fs)) / target_fs
    x = np.atleast_2d(x)
    return np.stack([np.interp(grid, t, row) for row in x])


def dct_basis(n: int, dt: float, cutoff: float = DRIFT_CUTOFF) -> np.ndarray:
    """Discrete cosine drift terms up to `cutoff` Hz, excluding the constant."""
    k = np.arange(1, int(np.floor(2 * n * dt * cutoff)) + 1)
    if len(k) == 0:
        return np.zeros((n, 0))
    t = np.arange(n)[:, None]
    return np.cos(np.pi * (2 * t + 1) * k[None, :] / (2 * n))


def regress_out(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residualise columns of `y` on `z`, with an intercept. Weights are shared
    across states by construction, which is the whole point of doing this here
    rather than handing the nuisance regressors to the HMM as inputs."""
    if z.shape[1] == 0:
        return y - y.mean(axis=0)
    design = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    sd = x.std(axis=axis, keepdims=True)
    return (x - x.mean(axis=axis, keepdims=True)) / np.where(sd > 0, sd, 1.0)


def prepare_subject(
    sub: str,
    variant: str = "ols",
    target_fs: float = TARGET_FS,
    window_s: float = WINDOW_S,
    band=BAND,
    nuisance: bool = True,
    temppar: str = "separate",
    deriv_root: Path | None = None,
    dataset_root: Path | None = None,
) -> list[RunData]:
    """Features and design for every usable run of one subject.

    Runs whose neural recording does not cover `window_s` from the first trial
    onset are dropped with a warning rather than silently truncated.
    """
    runs = io.load_parcel_ts(sub, variant, deriv_root=deriv_root)
    events = io.load_all_events(sub, dataset_root=dataset_root)
    if len(runs) != len(events):
        raise ValueError(f"{sub}: {len(runs)} neural runs but {len(events)} event tables")

    parcels = [str(p) for p in runs[0].coords["parcel"].values]
    net = network_labels(parcels, temppar)
    mask = covered_mask(runs)

    networks = [n for n in FEATURE_NETWORKS if not (temppar == "default" and n == "TempPar")]
    empty = [n for n in networks if not (mask & (net == n)).any()]
    if empty:
        raise ValueError(f"{sub}: no covered parcels in {empty}")

    feature_names = [f"{n}_{c}" for c in CHROMOPHORES for n in networks]
    out = []
    for run_idx, (da, ev) in enumerate(zip(runs, events, strict=True), start=1):
        t = _values(da.coords["time"])
        fs = 1.0 / np.median(np.diff(t))
        t0 = float(ev["onset"].min())
        if t0 + window_s > t[-1]:
            warnings.warn(
                f"{sub} run {run_idx}: neural run ends {t[-1] - t0:.0f} s after the first "
                f"onset, short of the {window_s:.0f} s window; dropping the run",
                stacklevel=2,
            )
            continue

        vals = _values(da)  # (chromo, parcel, time)
        raw = np.stack(
            [
                vals[list(CHROMOPHORES).index(c)][mask & (net == n)].mean(axis=0)
                for c in CHROMOPHORES
                for n in networks
            ]
        )
        # Global signal is the mean over every covered parcel, not just the ones
        # that feed a feature, so it stays a whole-head nuisance term.
        glob = np.stack([vals[i][mask].mean(axis=0) for i in range(len(CHROMOPHORES))])

        y = resample_window(bandpass(raw, fs, band), t, t0, window_s, target_fs).T
        g = resample_window(bandpass(glob, fs, band), t, t0, window_s, target_fs).T
        x = resample_window(bandpass(design_matrix(ev, t, fs).T, fs, band), t, t0, window_s,
                            target_fs).T  # fmt: skip

        if nuisance:
            z = np.column_stack(
                [g, np.gradient(g, axis=0), dct_basis(len(y), 1.0 / target_fs)]
            )
            # Residualise the inputs too, so the task weights match what a joint
            # fit with shared nuisance weights would have given.
            y, x = regress_out(y, z), regress_out(x, z)

        out.append(
            RunData(
                sub=sub,
                run=run_idx,
                y=zscore(y),
                x=zscore(x),
                t0=t0,
                target_fs=target_fs,
                n_covered=int(mask.sum()),
                feature_names=feature_names,
                input_names=list(INPUT_NAMES),
                events=ev,
            )
        )
    return out


def prepare_all(subs=None, verbose: bool = True, **kwargs) -> list[RunData]:
    """Every usable run of every subject, ready to stack into a group fit."""
    subs = subs or io.subject_ids(kwargs.get("variant", "ols"))
    runs = []
    for sub in subs:
        got = prepare_subject(sub, **kwargs)
        runs.extend(got)
        if verbose:
            print(f"{sub}: {len(got)} runs, {got[0].n_covered if got else 0} covered parcels")
    return runs


def stack(runs: list[RunData]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Batch runs into (N, T, D) emissions, (N, T, M) inputs and a row index."""
    y = np.stack([r.y for r in runs])
    x = np.stack([r.x for r in runs])
    meta = pd.DataFrame([{"sub": r.sub, "run": r.run, "n_covered": r.n_covered} for r in runs])
    return y, x, meta


def trial_samples(run: RunData, lag_s: float = 0.0) -> np.ndarray:
    """Neural sample index per trial, -1 for trials outside the window.

    `lag_s` shifts the lookup forward in time to meet the haemodynamic response,
    so a trial at onset o is scored against the neural sample at o + lag.
    """
    idx = np.rint((run.events["onset"].to_numpy(float) - run.t0 + lag_s) * run.target_fs)
    idx = idx.astype(int)
    return np.where((idx >= 0) & (idx < run.n_samples), idx, -1)


def save_runs(runs: list[RunData], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(runs, f)
    return path


def load_runs(path: Path) -> list[RunData]:
    with open(path, "rb") as f:
        return pickle.load(f)
