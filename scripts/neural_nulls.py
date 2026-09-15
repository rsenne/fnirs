"""Compare neural GLM-HMM results with surrogate data and shifted state sequences.

Smooth, autocorrelated signals can produce apparent HMM states. These comparisons
ask which fitted patterns also occur under three null procedures:

- AR surrogates simulate residuals with per-feature autoregressive models and
  shared innovation covariance, then add the fitted task response. Simulations
  are bandpassed to match the preprocessing of the real features.
- Phase-randomised surrogates preserve the residuals' power spectra and
  cross-spectra while changing their temporal arrangement.
- Circular shifts move each run's fitted state probabilities relative to its
  trials, testing their alignment with behaviour.

Surrogate fits and cross-validation use the saved state count for each subject;
they do not repeat the search over K. Circular shifts reuse the real fits.

Reads results from scripts/fit_neural_hmm.py and writes to results/neural/.
Run with `uv run python scripts/neural_nulls.py`.
"""

import os

# Set thread limits before importing JAX.
NSLOTS = int(os.environ.get("NSLOTS", "8"))
WORKERS = max(1, min(NSLOTS, 12))
THREADS = max(1, NSLOTS // WORKERS)
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fnirs_glmhmm import config, coupling, neural  # noqa: E402
from fnirs_glmhmm.model import fit_glm_hmm, no_state_log_prob  # noqa: E402

from fit_neural_hmm import (  # noqa: E402  isort: skip
    REPORT_LAG,
    posteriors,
    sort_states,
)

OUTCOMES = ("commission", "omission", "log_rt", "vtc")
NULL_PREDICTORS = ("p_dmn_high", "p_dmn_low")


def fit_ar(x: np.ndarray, max_p: int = 5) -> tuple[np.ndarray, float]:
    """Least-squares AR fit to one feature, order picked by AIC.

    Returns the selected coefficients and the innovation variance.
    """
    best = None
    for p in range(1, max_p + 1):
        design = np.column_stack([x[p - j - 1 : len(x) - j - 1] for j in range(p)])
        target = x[p:]
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ coef
        sigma2 = float(resid.var())
        aic = len(target) * np.log(sigma2) + 2 * p
        if best is None or aic < best[0]:
            best = (aic, coef, sigma2)
    return best[1], best[2]


def ar_model(resid_runs: list[np.ndarray], max_p: int = 5):
    """Per-feature AR coefficients plus the cross-feature innovation covariance.

    Each feature gets its own temporal model. The innovation covariance captures
    correlations between the models' residuals.
    """
    stacked = np.concatenate(resid_runs)
    coefs = [fit_ar(stacked[:, d], max_p)[0] for d in range(stacked.shape[1])]
    order = max(len(c) for c in coefs)

    innov = np.empty((len(stacked) - order, stacked.shape[1]))
    for d, c in enumerate(coefs):
        pred = sum(c[j] * stacked[order - j - 1 : len(stacked) - j - 1, d] for j in range(len(c)))
        innov[:, d] = stacked[order:, d] - pred
    return coefs, np.cov(innov, rowvar=False)


def simulate_ar(coefs, cov, n_samples: int, rng, burn: int = 300) -> np.ndarray:
    order = max(len(c) for c in coefs)
    n = n_samples + burn
    e = rng.multivariate_normal(np.zeros(len(coefs)), cov, size=n)
    y = np.zeros_like(e)
    for t in range(order, n):
        for d, c in enumerate(coefs):
            y[t, d] = e[t, d] + sum(c[j] * y[t - j - 1, d] for j in range(len(c)))
    return y[burn:]


def phase_surrogate(resid: np.ndarray, rng) -> np.ndarray:
    """Randomise residual phases while preserving spectra within a run.

    Applying the same random phase to every feature at each frequency preserves
    the power spectra and pairwise cross-spectra. This avoids the spectral
    approximation made by the finite-order AR model.
    """
    n = len(resid)
    spectrum = np.fft.rfft(resid, axis=0)
    phases = rng.uniform(0, 2 * np.pi, spectrum.shape[0])
    phases[0] = 0.0  # keep the mean real
    if n % 2 == 0:
        phases[-1] = 0.0  # and the Nyquist bin
    return np.fft.irfft(spectrum * np.exp(1j * phases)[:, None], n=n, axis=0)


def surrogate_runs(runs, rng, kind: str = "ar", max_p: int = 5, pad: int = 300):
    """Add surrogate residuals to the fitted task response, then standardise."""
    y = np.stack([r.y for r in runs])
    x = np.stack([r.x for r in runs])
    design = [np.column_stack([np.ones(len(r.x)), r.x]) for r in runs]

    flat_d = np.concatenate(design)
    beta, *_ = np.linalg.lstsq(flat_d, np.concatenate(list(y)), rcond=None)
    resid = [yi - d @ beta for yi, d in zip(y, design, strict=True)]

    if kind == "phase":
        noise = [phase_surrogate(r, rng) for r in resid]
    elif kind == "ar":
        coefs, cov = ar_model(resid, max_p)
        # Simulate long and filter before trimming: the features were bandpassed
        # before the analysis window was cut, and an unfiltered AR draw carries
        # low-frequency power the real features do not have.
        fs = runs[0].target_fs
        noise = []
        for d in design:
            sim = simulate_ar(coefs, cov, len(d) + 2 * pad, rng)
            noise.append(neural.bandpass(sim.T, fs)[:, pad : pad + len(d)].T)
    else:
        raise ValueError(f"unknown surrogate kind {kind!r}")

    return [neural.zscore(d @ beta + e) for d, e in zip(design, noise, strict=True)], x


def cv_gain(y, x, k, restarts, iters) -> float:
    """Held-out log-likelihood gain over one-state regression, in nats per sample."""
    gains = []
    for fold in range(len(y)):
        test = np.zeros(len(y), dtype=bool)
        test[fold] = True
        fit = fit_glm_hmm(
            y[~test], x[~test], num_states=k, kind="gaussian_diag",
            num_restarts=restarts, num_iters=iters, init_method="kmeans",
        )  # fmt: skip
        base = no_state_log_prob(y[~test], x[~test], y[test], x[test])
        gains.append((float(fit.marginal_log_prob(y[test], x[test]).sum()) - base) / y.shape[1])
    return float(np.mean(gains))


def coupling_slopes(runs, post, lag: float = REPORT_LAG) -> dict:
    """The behavioural statistics, keyed `outcome:predictor`."""
    frame = coupling.coupling_frame(runs, list(post), lag_s=lag)
    frame["p_dmn_low"] = frame["p0"]
    frame["p_dmn_high"] = frame[f"p{post.shape[-1] - 1}"]
    rows = coupling.subject_coupling(frame, NULL_PREDICTORS, OUTCOMES)
    return {f"{r.outcome}:{r.predictor}": r.slope for r in rows.itertuples()}


def baseline_contrast(biases: np.ndarray, feature_names) -> dict:
    """DMN-high state minus DMN-low state, per feature.

    States are sorted on Default_HbO, so that feature's contrast is positive by
    construction. Contrasts in other networks can be compared with surrogates
    to assess how much of the pattern is explained by their covariance.
    """
    diff = biases[-1] - biases[0]
    return {f"contrast:{n}": float(v) for n, v in zip(feature_names, diff, strict=True)}


def shift_posteriors(post: np.ndarray, rng) -> np.ndarray:
    """Roll each run's state posterior by its own random offset."""
    out = np.empty_like(post)
    for i in range(len(post)):
        out[i] = np.roll(post[i], int(rng.integers(1, post.shape[1])), axis=0)
    return out


def run_subject(sub: str, args) -> pd.DataFrame | None:
    fit_path = args.outdir / f"fit_{sub}_{args.tag}.npz"
    if not fit_path.exists():
        print(f"{sub}: no fit at {fit_path}, skipping", flush=True)
        return None
    blob = np.load(fit_path, allow_pickle=False)
    k = int(blob["best_k"])

    runs = neural.prepare_subject(
        sub, variant=args.variant, target_fs=args.target_fs, nuisance=not args.no_nuisance
    )
    y, x, _ = neural.stack(runs)
    # Deterministic per subject: hash() is salted per process under spawn.
    rng = np.random.default_rng(args.seed + int(re.sub(r"\D", "", sub)))

    rows = []
    real_post = blob["posteriors"]
    feature_names = [str(f) for f in blob["feature_names"]]
    rows.append(
        dict(sub=sub, k=k, draw=-1, kind="real",
             cv_gain=cv_gain(y, x, k, args.restarts, args.iters),
             **baseline_contrast(blob["biases"], feature_names),
             **coupling_slopes(runs, real_post))
    )  # fmt: skip

    # Circular shift: real states, wrong time. Cheap, so it gets the draws.
    for draw in range(args.shift_draws):
        rows.append(
            dict(sub=sub, k=k, draw=draw, kind="shift", cv_gain=np.nan,
                 **coupling_slopes(runs, shift_posteriors(real_post, rng)))
        )  # fmt: skip

    # Refit and cross-validate surrogate data at the real fit's selected K.
    for surrogate in args.surrogates:
        for draw in range(args.ar_draws):
            sim, xs = surrogate_runs(runs, rng, kind=surrogate)
            ysim = np.stack(sim)
            fit = fit_glm_hmm(
                ysim, xs, num_states=k, kind="gaussian_diag", num_restarts=args.restarts,
                num_iters=args.iters, init_method="kmeans",
            )  # fmt: skip
            order = sort_states(fit, feature_names)
            biases = np.asarray(fit.params.emissions.biases)[order]
            rows.append(
                dict(sub=sub, k=k, draw=draw, kind=surrogate,
                     cv_gain=cv_gain(ysim, xs, k, args.restarts, args.iters),
                     **baseline_contrast(biases, feature_names),
                     **coupling_slopes(runs, posteriors(fit, ysim, xs, order)))
            )  # fmt: skip

    import jax

    jax.clear_caches()
    print(
        f"{sub}: {args.ar_draws} draws of {'+'.join(args.surrogates)}, "
        f"{args.shift_draws} shift draws",
        flush=True,
    )
    return pd.DataFrame(rows)


ONE_SIDED = ("cv_gain",)  # a likelihood gain only counts if it is larger, not merely different


def summarize(nulls: pd.DataFrame, stats: list[str], n_resamples: int = 2000, seed: int = 0):
    """Where the real statistic sits in each null, per subject and for the group.

    Each group resample takes one null draw per subject and averages them.
    Resampling combinations avoids requiring draw indices to match across subjects.
    The resulting comparison still depends on the available draws per subject.
    """
    rng = np.random.default_rng(seed)
    out = []
    real = nulls[nulls.kind == "real"].set_index("sub")
    for kind in [k for k in nulls.kind.unique() if k != "real"]:
        draws = nulls[nulls.kind == kind]
        if draws.empty:
            continue
        for stat in stats:
            if stat not in draws or draws[stat].isna().all():
                continue
            per_sub_p, real_vals, columns = [], [], []
            for sub, g in draws.groupby("sub"):
                v = g[stat].dropna().to_numpy()
                r = real.loc[sub, stat] if sub in real.index else np.nan
                if not len(v) or not np.isfinite(r):
                    continue
                one = stat in ONE_SIDED
                per_sub_p.append(np.mean(v >= r) if one else np.mean(abs(v) >= abs(r)))
                real_vals.append(r)
                columns.append(v)
            if not columns:
                continue

            # Draw one statistic per subject to form each group null mean.
            picks = np.stack([rng.choice(c, size=n_resamples) for c in columns])
            group_null = picks.mean(axis=0)
            real_mean = float(np.mean(real_vals))
            p = (
                np.mean(group_null >= real_mean)
                if stat in ONE_SIDED
                else np.mean(np.abs(group_null) >= abs(real_mean))
            )
            out.append(
                dict(
                    kind=kind,
                    statistic=stat,
                    real_mean=real_mean,
                    null_mean=float(group_null.mean()),
                    null_sd=float(group_null.std()),
                    group_p=float(p),
                    n_subjects_p05=int(sum(pv < 0.05 for pv in per_sub_p)),
                    n_subjects=len(per_sub_p),
                )
            )
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-draws", type=int, default=12, help="draws per surrogate kind")
    ap.add_argument(
        "--surrogates",
        nargs="+",
        default=["ar", "phase"],
        choices=("ar", "phase"),
        help="phase randomisation matches the spectrum exactly; AR only approximates it",
    )
    ap.add_argument("--shift-draws", type=int, default=500)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--variant", default="ols")
    ap.add_argument("--target-fs", type=float, default=neural.TARGET_FS)
    ap.add_argument("--no-nuisance", action="store_true")
    ap.add_argument("--subs", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--outdir", type=Path, default=config.RESULTS_DIR / "neural")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.tag = args.tag or ("nonuis" if args.no_nuisance else "main")

    subs = args.subs or sorted(
        p.name.split("_")[1] for p in args.outdir.glob(f"fit_sub-*_{args.tag}.npz")
    )
    print(f"{len(subs)} subjects, {args.workers} workers x {THREADS} threads")

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        frames = [f for f in pool.map(run_subject, subs, [args] * len(subs)) if f is not None]

    nulls = pd.concat(frames, ignore_index=True)
    nulls.to_csv(args.outdir / f"nulls_{args.tag}.csv", index=False)

    stats = ["cv_gain"] + [c for c in nulls.columns if ":" in c]
    summary = summarize(nulls, stats, seed=args.seed)
    summary.to_csv(args.outdir / f"nulls_summary_{args.tag}.csv", index=False)
    print()
    print(summary.round(4).to_string(index=False))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
