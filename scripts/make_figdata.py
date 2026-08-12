"""Precompute everything the figures need, so plotting scripts never refit models.

Writes tidy CSVs to results/figdata/. Run before scripts/figures/*.py.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fnirs_glmhmm import config, io  # noqa: E402
from fnirs_glmhmm.behavior import smooth_vtc, zone_labels  # noqa: E402
from fnirs_glmhmm.zones import fit_zone_hmm  # noqa: E402

OUT = config.RESULTS_DIR / "figdata"
MAXLAG = 120
PERI = 10
TRIAL_S = 0.8


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = np.dot(x, x)
    return np.array([np.dot(x[: len(x) - k], x[k:]) / d for k in range(nlags + 1)])


def per_subject(sub):
    """Everything that needs a model fit or per-trial alignment, for one subject."""
    ev = io.load_all_events(sub)
    vtc = [e.VTC.to_numpy() for e in ev]

    fit_raw = fit_zone_hmm(vtc, num_states=2, num_restarts=8, num_iters=200)
    fit_sm = fit_zone_hmm(
        [smooth_vtc(v) for v in vtc], num_states=2, transform="zscore",
        num_restarts=8, num_iters=200,
    )  # fmt: skip

    allev = pd.concat(ev, ignore_index=True)
    go = (allev.trial_type == "city").to_numpy()
    omit = (allev.response_code == -1).to_numpy()

    masks = {
        "HMM (raw VTC)": np.concatenate([s == 1 for s in fit_raw.states()]),
        "HMM (smoothed)": np.concatenate([s == 1 for s in fit_sm.states()]),
        "Median split": np.concatenate([zone_labels(smooth_vtc(v))[0].astype(bool) for v in vtc]),
    }
    lift = {}
    for name, m in masks.items():
        i_, o_ = go & ~m, go & m
        lift[name] = (omit[o_].mean() if o_.any() else np.nan) - (
            omit[i_].mean() if i_.any() else np.nan
        )

    # Autocorrelation, one curve per run.
    acfs = [(acf(v, MAXLAG), acf(smooth_vtc(v), MAXLAG)) for v in vtc]

    # Peri-error VTC, responded trials only so no interpolated values enter.
    peri = []
    for e in ev:
        v, rc = e.VTC.to_numpy(), e.response_code.to_numpy()
        resp = e.reaction_time.to_numpy() > 0
        n = len(v)
        for kind, idx in (
            ("omission", np.flatnonzero(rc == -1)),
            ("commission", np.flatnonzero(rc == -2)),
        ):
            for t in idx:
                for lag in range(-PERI, PERI + 1):
                    j = t + lag
                    if 0 <= j < n and resp[j]:
                        peri.append((kind, lag, v[j]))
    peri = pd.DataFrame(peri, columns=["kind", "lag", "vtc"])
    peri = peri.groupby(["kind", "lag"], as_index=False).vtc.mean()
    peri["sub"] = sub
    peri["baseline"] = float(np.concatenate(vtc).mean())

    import jax

    jax.clear_caches()

    return dict(
        sub=sub,
        lift=lift,
        acfs=acfs,
        peri=peri,
        dwell_in=float(fit_raw.expected_dwell[0]),
        dwell_out=float(fit_raw.expected_dwell[1]),
        dwell_in_sm=float(fit_sm.expected_dwell[0]),
        dwell_out_sm=float(fit_sm.expected_dwell[1]),
        switches_raw=float(
            np.mean([np.diff((s == 1).astype(int)).astype(bool).sum() for s in fit_raw.states()])
        ),
        switches_split=float(
            np.mean([np.diff(zone_labels(smooth_vtc(v))[0]).astype(bool).sum() for v in vtc])
        ),
    )


def example_trace(sub="sub-629", run_ix=0):
    """One run, both segmentations, for the illustrative figure."""
    ev = io.load_all_events(sub)
    vtc = [e.VTC.to_numpy() for e in ev]
    fit = fit_zone_hmm(vtc, num_states=2, num_restarts=8, num_iters=200)
    v = vtc[run_ix]
    return pd.DataFrame(
        dict(
            trial=np.arange(len(v)),
            time_s=np.arange(len(v)) * TRIAL_S,
            vtc=v,
            vtc_smoothed=smooth_vtc(v),
            hmm_state=fit.states()[run_ix],
            hmm_p_out=fit.out_of_zone_prob()[run_ix],
            split_out=zone_labels(smooth_vtc(v))[0],
            omission=(ev[run_ix].response_code == -1).to_numpy().astype(int),
            commission=(ev[run_ix].response_code == -2).to_numpy().astype(int),
        )
    )


def error_vtc_table():
    """Neighbourhood VTC for errors vs correct trials, the dissociation figure."""
    W = 5
    rows = []
    for sub in io.subject_ids():
        for e in io.load_all_events(sub):
            v = e.VTC.to_numpy()
            resp = e.reaction_time.to_numpy() > 0
            n = len(v)
            loc = np.full(n, np.nan)
            for t in range(n):
                idx = np.arange(max(0, t - W), min(n, t + W + 1))
                idx = idx[(idx != t) & resp[idx]]
                if len(idx):
                    loc[t] = v[idx].mean()
            rows.append(pd.DataFrame(dict(sub=sub, loc=loc, tt=e.trial_type, rc=e.response_code)))
    d = pd.concat(rows, ignore_index=True).dropna(subset=["loc"])
    out = []
    for kind, sel, errcode in (
        ("commission", d.tt == "mnt", -2),
        ("omission", d.tt == "city", -1),
    ):
        s = d[sel]
        g = s.assign(err=s.rc == errcode).groupby(["sub", "err"]).loc.mean().unstack()
        g = g.dropna().rename(columns={False: "correct", True: "error"})
        out.append(g.reset_index().assign(kind=kind))
    return pd.concat(out, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subs = io.subject_ids()

    with ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context("spawn")) as pool:
        res = [f.result() for f in [pool.submit(per_subject, s) for s in subs]]

    # 1. autocorrelation
    raw = np.array([a for r in res for a, _ in r["acfs"]])
    sm = np.array([b for r in res for _, b in r["acfs"]])
    lags = np.arange(MAXLAG + 1)
    pd.DataFrame(
        dict(
            lag=np.tile(lags, 2),
            lag_s=np.tile(lags * TRIAL_S, 2),
            series=np.repeat(["Raw VTC", "Smoothed (L=20)"], len(lags)),
            acf=np.concatenate([raw.mean(0), sm.mean(0)]),
            se=np.concatenate([raw.std(0) / np.sqrt(len(raw)), sm.std(0) / np.sqrt(len(sm))]),
        )
    ).to_csv(OUT / "acf.csv", index=False)

    # 2. per-subject zone measure comparison
    lift = pd.DataFrame([{"sub": r["sub"], **r["lift"]} for r in res])
    lift.to_csv(OUT / "omission_lift.csv", index=False)

    # 3. dwell times and switch counts
    pd.DataFrame(
        [
            {
                k: r[k]
                for k in (
                    "sub",
                    "dwell_in",
                    "dwell_out",
                    "dwell_in_sm",
                    "dwell_out_sm",
                    "switches_raw",
                    "switches_split",
                )  # fmt: skip
            }
            for r in res
        ]
    ).to_csv(OUT / "dynamics.csv", index=False)

    # 4. peri-error VTC
    pd.concat([r["peri"] for r in res], ignore_index=True).to_csv(
        OUT / "peri_error.csv", index=False
    )

    # 5. error vs correct neighbourhood VTC
    error_vtc_table().to_csv(OUT / "error_vtc.csv", index=False)

    # 6. example trace
    example_trace().to_csv(OUT / "example_trace.csv", index=False)

    # 7. BIC by state count, from the earlier sweep
    src = config.RESULTS_DIR / "zone_hmm_vs_split.csv"
    if src.exists():
        m = pd.read_csv(src)
        m[["sub", "bic2", "bic3", "bic4", "bic5", "best_k"]].to_csv(OUT / "bic.csv", index=False)

    print(f"wrote {len(list(OUT.glob('*.csv')))} files to {OUT}")
    for f in sorted(OUT.glob("*.csv")):
        print(f"  {f.name:20s} {len(pd.read_csv(f)):6d} rows")


if __name__ == "__main__":
    main()
