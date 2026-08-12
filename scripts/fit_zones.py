"""Fit zone HMMs for every subject and check them against the median split.

Scores both measures against commission errors, which VTC has no relationship to in
this dataset, so that comparison comes out null for everything. The BIC sweep and
the descriptive columns are what still get used; they feed results/figdata/bic.csv.
Omission scoring lives in scripts/make_figdata.py.

Run with `uv run python scripts/fit_zones.py`. One process per subject.
"""

import os

# Has to happen before jax is imported anywhere.
NSLOTS = int(os.environ.get("NSLOTS", "8"))
WORKERS = max(1, min(NSLOTS, 12))
THREADS = max(1, NSLOTS // WORKERS)
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fnirs_glmhmm import config, io  # noqa: E402
from fnirs_glmhmm.behavior import smooth_vtc, zone_labels  # noqa: E402
from fnirs_glmhmm.zones import compare_to_median_split, fit_zone_hmm  # noqa: E402

STATE_RANGE = (2, 3, 4, 5)


def commission_rate(events: pd.DataFrame, out_mask: np.ndarray) -> tuple[float, float, int]:
    """Commission error rate on no-go trials, split by zone.

    Returns (rate_in, rate_out, n_nogo). NaN where a zone has no no-go trials.
    """
    nogo = (events["trial_type"] == "mnt").to_numpy()
    err = (events["response_code"] == -2).to_numpy()
    inz, outz = nogo & ~out_mask.astype(bool), nogo & out_mask.astype(bool)
    r_in = err[inz].mean() if inz.any() else np.nan
    r_out = err[outz].mean() if outz.any() else np.nan
    return float(r_in), float(r_out), int(nogo.sum())


def run_subject(sub: str, num_restarts: int = 10, num_iters: int = 200) -> dict:
    events = io.load_all_events(sub)
    vtc = [e["VTC"].to_numpy() for e in events]

    fit = fit_zone_hmm(vtc, num_states=2, num_restarts=num_restarts, num_iters=num_iters)
    cmp = compare_to_median_split(fit, vtc)

    # Pool trials across runs before computing error rates; no-go trials are rare.
    hmm_out = np.concatenate([(s == 1).astype(int) for s in fit.states()])
    split_out = np.concatenate([zone_labels(smooth_vtc(v))[0] for v in vtc])
    allev = pd.concat(events, ignore_index=True)

    h_in, h_out, n_nogo = commission_rate(allev, hmm_out)
    s_in, s_out, _ = commission_rate(allev, split_out)

    bics = {}
    for k in STATE_RANGE:
        f = fit_zone_hmm(vtc, num_states=k, num_restarts=6, num_iters=num_iters)
        bics[k] = f.bic()

    import jax

    jax.clear_caches()

    return dict(
        sub=sub,
        n_runs=len(vtc),
        n_trials=len(allev),
        n_nogo=n_nogo,
        frac_out_hmm=cmp["frac_out_hmm"],
        frac_out_split=cmp["frac_out_median_split"],
        agreement=cmp["agreement"],
        switches_hmm=float(np.mean(cmp["switches_hmm"])),
        switches_split=float(np.mean(cmp["switches_median_split"])),
        dwell_in=fit.expected_dwell[0],
        dwell_out=fit.expected_dwell[1],
        mean_in=fit.state_means[0],
        mean_out=fit.state_means[1],
        comm_in_hmm=h_in,
        comm_out_hmm=h_out,
        comm_in_split=s_in,
        comm_out_split=s_out,
        best_k=min(bics, key=bics.get),
        **{f"bic{k}": v for k, v in bics.items()},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    subs = io.subject_ids()
    print(f"{len(subs)} subjects, {args.workers} workers x {THREADS} threads")

    # Spawn, not fork: this module imports jax, and forking a process with XLA's
    # thread pools already up deadlocks every child before it does any work.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_subject, s, args.restarts, args.iters) for s in subs]
        rows = []
        for fut in futures:
            rows.append(fut.result())
            print(".", end="", flush=True)
    print()

    df = pd.DataFrame(rows).sort_values("sub").reset_index(drop=True)
    df["comm_lift_hmm"] = df.comm_out_hmm - df.comm_in_hmm
    df["comm_lift_split"] = df.comm_out_split - df.comm_in_split

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "zone_hmm_vs_split.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    show = [
        "sub", "frac_out_hmm", "frac_out_split", "agreement",
        "switches_hmm", "switches_split", "dwell_in", "dwell_out",
        "comm_in_hmm", "comm_out_hmm", "comm_in_split", "comm_out_split", "best_k",
    ]  # fmt: skip
    print(df[show].round(3).to_string(index=False))
    print(f"\nwrote {out}")

    print("\n--- medians across subjects ---")
    print(df[[c for c in df.columns if c != "sub"]].median().round(3).to_string())

    print("\n--- BIC-preferred state count ---")
    print(df.best_k.value_counts().sort_index().to_string())

    print("\n--- commission errors out minus in zone ---")
    for label in ("hmm", "split"):
        lift = df[f"comm_lift_{label}"].dropna()
        print(
            f"{label:6s} mean {lift.mean():+.4f}  median {lift.median():+.4f}  "
            f"positive in {int((lift > 0).sum())}/{len(lift)} subjects"
        )


if __name__ == "__main__":
    main()
