"""Loading parcel timeseries and behaviour off the cluster."""

import pickle
import re
from pathlib import Path

import pandas as pd
import xarray as xr

from fnirs_glmhmm import config


def subject_ids(deriv_root: Path | None = None) -> list[str]:
    """All sub-xx directories that actually exist, sorted numerically."""
    root = Path(deriv_root or config.DERIV_ROOT)
    subs = [p.name for p in root.glob("sub-*") if p.is_dir()]
    return sorted(subs, key=lambda s: int(re.sub(r"\D", "", s) or 0))


def parcel_path(sub: str, variant: str = "ols", deriv_root: Path | None = None) -> Path:
    if variant not in config.VARIANTS:
        raise ValueError(f"variant must be one of {config.VARIANTS}, got {variant!r}")
    root = Path(deriv_root or config.DERIV_ROOT)
    return root / sub / config.PARCEL_FILE_TEMPLATE.format(sub=sub, variant=variant)


def load_parcel_ts(
    sub: str,
    variant: str = "ols",
    drop_non_cortical: bool = True,
    deriv_root: Path | None = None,
) -> list[xr.DataArray]:
    """One DataArray per run, dims (chromophore, parcel, time).

    The pickle also carries `vertex_mse`, which we ignore.
    """
    with open(parcel_path(sub, variant, deriv_root), "rb") as f:
        blob = pickle.load(f)

    runs = blob["parcel_ts"]
    if drop_non_cortical:
        runs = [drop_non_cortical_parcels(r) for r in runs]
    return runs


def drop_non_cortical_parcels(da: xr.DataArray) -> xr.DataArray:
    """Strip medial wall / scalp so only the Schaefer parcels remain."""
    parcels = [str(p) for p in da.coords["parcel"].values]
    keep = [p for p in parcels if p not in config.NON_CORTICAL_PARCELS]
    return da.sel(parcel=keep)


def events_path(sub: str, run: int, dataset_root: Path | None = None) -> Path:
    root = Path(dataset_root or config.DATASET_ROOT)
    return root / sub / "nirs" / f"{sub}_task-gradCPT_run-{run:02d}_events.tsv"


def load_events(sub: str, run: int, dataset_root: Path | None = None) -> pd.DataFrame:
    """Trial table with at least `onset` and `VTC`."""
    return pd.read_csv(events_path(sub, run, dataset_root), sep="\t")


def load_all_events(sub: str, dataset_root: Path | None = None) -> list[pd.DataFrame]:
    """Every run's event table, ordered by run number."""
    root = Path(dataset_root or config.DATASET_ROOT)
    paths = sorted((root / sub / "nirs").glob(f"{sub}_task-gradCPT_run-*_events.tsv"))
    return [pd.read_csv(p, sep="\t") for p in paths]


def load_eyetracking(sub: str, run: int, dataset_root: Path | None = None) -> pd.DataFrame:
    root = Path(dataset_root or config.DATASET_ROOT)
    path = (
        root / sub / "nirs" / f"{sub}_task-gradCPT_run-{run:02d}_recording-eyetracking_physio.tsv"
    )
    return pd.read_csv(path, sep="\t")
