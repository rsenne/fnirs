"""Per-subject GLM-HMMs on the fNIRS network timeseries, with the task as input.

One model per subject: their runs share parameters, nobody else's data enters the
fit, and K is chosen for each subject by leaving out one of their own runs. States
are brain states; behaviour never enters the fit and is scored afterwards as
held-out validation.

Because every subject has their own states, the group claim is a second-level test
over per-subject effects, not a pooled model. States are aligned across subjects
only by their DMN baseline: the highest- and lowest-Default_HbO states are what get
carried up to the group level.

Writes to results/neural/. Run with `uv run python scripts/fit_neural_hmm.py`.
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fnirs_glmhmm import config, coupling, io, neural  # noqa: E402
from fnirs_glmhmm.model import fit_glm_hmm, no_state_log_prob  # noqa: E402

SORT_FEATURE = "Default_HbO"  # states are labelled by ascending DMN baseline
LAGS = (2.0, 4.0, 6.0, 8.0)
REPORT_LAG = 4.0
PREDICTORS = ("dmn_score", "p_dmn_high", "p_dmn_low", "p_out_zone", "out_of_zone_split")


def sort_states(fit, feature_names: list[str], key: str = SORT_FEATURE) -> np.ndarray:
    """Permutation putting states in ascending order of their baseline in `key`.

    Without it the labels are whatever EM landed on, and nothing is comparable
    across refits, let alone across subjects.
    """
    biases = np.asarray(fit.params.emissions.biases)
    return np.argsort(biases[:, feature_names.index(key)])


def posteriors(fit, y: np.ndarray, x: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Smoothed state probabilities per run, (N, T, K), in the sorted labelling."""
    out = [np.asarray(fit.posterior(y[i], inputs=x[i]).smoothed_probs) for i in range(len(y))]
    return np.stack(out)[..., order]


def choose_k(y, x, ks, restarts, iters, kind: str) -> pd.DataFrame:
    """Leave out one run, refit, score it. One row per (k, held-out run).

    k = 1 is the no-states baseline: a single linear regression, fit in closed form
    because dynamax cannot build a one-state HMM. Everything else has to beat it.
    """
    rows = []
    for fold in range(len(y)):
        test = np.zeros(len(y), dtype=bool)
        test[fold] = True
        rows.append(
            dict(
                k=1,
                fold=fold,
                test_ll=no_state_log_prob(y[~test], x[~test], y[test], x[test]) / y.shape[1],
                train_ll=np.nan,
            )
        )
        for k in ks:
            fit = fit_glm_hmm(
                y[~test], x[~test], num_states=k, kind=kind,
                num_restarts=restarts, num_iters=iters, init_method="kmeans",
            )  # fmt: skip
            rows.append(
                dict(
                    k=k,
                    fold=fold,
                    test_ll=float(fit.marginal_log_prob(y[test], x[test]).sum() / y.shape[1]),
                    train_ll=float(fit.final_log_prob / ((~test).sum() * y.shape[1])),
                )
            )
    return pd.DataFrame(rows)


def state_profiles(fit, order, feature_names, input_names) -> pd.DataFrame:
    """Baseline b_k and task gains A_k, one row per state x feature."""
    b = np.asarray(fit.params.emissions.biases)[order]
    a = np.asarray(fit.params.emissions.weights)[order]
    k = len(order)
    rows = []
    for s in range(k):
        role = "dmn_high" if s == k - 1 else "dmn_low" if s == 0 else "middle"
        for d, name in enumerate(feature_names):
            net, chromo = name.rsplit("_", 1)
            row = dict(
                state=s, role=role, feature=name, network=net, chromo=chromo, baseline=b[s, d]
            )
            row.update({f"w_{n}": a[s, d, m] for m, n in enumerate(input_names)})
            rows.append(row)
    return pd.DataFrame(rows)


def hbo_hbr_signature(profiles: pd.DataFrame) -> pd.DataFrame:
    """Correlate the HbO and HbR halves of each state's baseline across networks.

    Negative is what a neurovascular state looks like. Positive means the state is
    tracking something systemic and must not be read as cognition.
    """
    rows = []
    for s, d in profiles.groupby("state"):
        wide = d.pivot(index="network", columns="chromo", values="baseline")
        r = float(np.corrcoef(wide["HbO"], wide["HbR"])[0, 1])
        rows.append(dict(state=int(s), r_hbo_hbr=r))
    return pd.DataFrame(rows)


def dynamics(fit, order, post) -> pd.DataFrame:
    p = fit.transition_matrix()[np.ix_(order, order)]
    return pd.DataFrame(
        dict(
            state=np.arange(len(order)),
            self_transition=np.diag(p),
            dwell_s=1.0 / np.clip(1.0 - np.diag(p), 1e-12, None),
            occupancy=post.reshape(-1, len(order)).mean(axis=0),
        )
    )


def run_subject(sub: str, args) -> dict | None:
    """Everything for one subject: K by CV, final fit, profiles, behaviour coupling."""
    runs = neural.prepare_subject(
        sub,
        variant=args.variant,
        target_fs=args.target_fs,
        nuisance=not args.no_nuisance,
    )
    if len(runs) < 2:
        print(f"{sub}: {len(runs)} usable run(s), skipping", flush=True)
        return None

    y, x, _ = neural.stack(runs)
    feature_names, input_names = runs[0].feature_names, runs[0].input_names
    ks = range(args.kmin, args.kmax + 1)

    cv = choose_k(y, x, ks, args.restarts, args.iters, args.kind)
    curve = cv.groupby("k")["test_ll"].mean()
    best_k = int(curve.drop(index=1).idxmax())
    # What the states buy over no states at all. Negative means this subject has no
    # state structure that survives cross-validation, and the fit below is descriptive.
    gain = float(curve[best_k] - curve[1])

    fit = fit_glm_hmm(
        y, x, num_states=best_k, kind=args.kind, num_restarts=args.restarts,
        num_iters=args.iters, init_method="kmeans",
    )  # fmt: skip
    order = sort_states(fit, feature_names)
    post = posteriors(fit, y, x, order)

    profiles = state_profiles(fit, order, feature_names, input_names)
    dyn = dynamics(fit, order, post)
    hbo_hbr = hbo_hbr_signature(profiles)

    # Expected DMN baseline at each sample: sum_k P(state k) b_k[Default_HbO]. Unlike a
    # single state's posterior it uses every state and stays on one scale whether the
    # subject took K = 2 or K = 8, which is what makes it comparable across people.
    dmn = np.asarray(fit.params.emissions.biases)[order][:, feature_names.index(SORT_FEATURE)]
    dmn_score = {r.run: post[i] @ dmn for i, r in enumerate(runs)}

    zone = coupling.zone_reference(sub)
    trials, coup = None, []
    for lag in args.lags:
        frame = coupling.coupling_frame(runs, list(post), lag_s=lag)
        # The only two states that mean the same thing for everybody.
        frame["p_dmn_low"] = frame["p0"]
        frame["p_dmn_high"] = frame[f"p{best_k - 1}"]
        frame["dmn_score"] = [
            dmn_score[r][s] for r, s in zip(frame["run"], frame["sample"], strict=True)
        ]
        frame["dmn_score"] = neural.zscore(frame["dmn_score"].to_numpy())
        frame = frame.merge(zone, on=["sub", "run", "trial"], how="left")
        rows = coupling.subject_coupling(frame, PREDICTORS)
        rows.insert(0, "lag_s", lag)
        coup.append(rows)
        if lag == args.report_lag:
            trials = frame

    np.savez_compressed(
        args.outdir / f"fit_{sub}_{args.tag}.npz",
        biases=np.asarray(fit.params.emissions.biases)[order],
        weights=np.asarray(fit.params.emissions.weights)[order],
        covs=np.asarray(fit.params.emissions.covs)[order],
        transitions=fit.transition_matrix()[np.ix_(order, order)],
        posteriors=post,
        feature_names=np.array(feature_names),
        input_names=np.array(input_names),
        run=np.array([r.run for r in runs]),
        best_k=best_k,
    )

    import jax

    jax.clear_caches()

    def stamp(d):
        # best_k, not k: the CV table carries its own k column for the sweep.
        return d.assign(sub=sub, best_k=best_k)

    print(f"{sub}: {len(runs)} runs, k={best_k}, cv gain over no states {gain:+.3f}", flush=True)
    return dict(
        summary=pd.DataFrame(
            [dict(n_runs=len(runs), n_covered=runs[0].n_covered, cv_gain=gain)]
        ).pipe(stamp),
        cv=stamp(cv),
        profiles=stamp(profiles),
        dynamics=stamp(dyn),
        hbo_hbr=stamp(hbo_hbr),
        coupling=stamp(pd.concat(coup, ignore_index=True)),
        trials=stamp(trials),
    )


def profile_contrast(profiles: pd.DataFrame) -> pd.DataFrame:
    """DMN-high state minus DMN-low state, per subject, then a group test.

    This is where the hypothesis lives: the high-DMN state should also show the
    blunted no-go gain.
    """
    quantities = [c for c in profiles.columns if c == "baseline" or c.startswith("w_")]
    high = profiles[profiles.role == "dmn_high"].set_index(["sub", "feature"])
    low = profiles[profiles.role == "dmn_low"].set_index(["sub", "feature"])
    diff = (high[quantities] - low[quantities]).reset_index()
    long = diff.melt(id_vars=["sub", "feature"], var_name="quantity", value_name="diff")
    return coupling.group_test(long, by=("feature", "quantity"), value="diff")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--variant", default="ols")
    ap.add_argument("--target-fs", type=float, default=neural.TARGET_FS)
    ap.add_argument(
        "--cov",
        choices=("diag", "full"),
        default="diag",
        help="full covariance is 105 params per state against ~700 training samples "
        "with lag-1 r near 0.95; it does not cross-validate",
    )
    ap.add_argument("--no-nuisance", action="store_true", help="skip global-signal regression")
    ap.add_argument("--subs", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--outdir", type=Path, default=config.RESULTS_DIR / "neural")
    ap.add_argument("--tag", default="", help="suffix on output filenames")
    args = ap.parse_args()

    args.tag = args.tag or ("nonuis" if args.no_nuisance else "main")
    args.kind = "gaussian_diag" if args.cov == "diag" else "gaussian"
    args.lags = LAGS
    args.report_lag = REPORT_LAG
    args.outdir.mkdir(parents=True, exist_ok=True)

    subs = args.subs or io.subject_ids(args.variant)
    print(f"{len(subs)} subjects, {args.workers} workers x {THREADS} threads")

    # Spawn, not fork: this module imports jax, and forking with XLA's thread pools
    # already up deadlocks every child.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        results = [r for r in pool.map(run_subject, subs, [args] * len(subs)) if r is not None]

    tables = {key: pd.concat([r[key] for r in results], ignore_index=True) for key in results[0]}
    for name, df in tables.items():
        df.to_csv(args.outdir / f"{name}_{args.tag}.csv", index=False)

    cv, coup = tables["cv"], tables["coupling"]
    print(f"\n{len(results)} subjects fit\n")
    print("--- held-out log likelihood, nats per sample, mean over subjects (k=1 is no states) ---")
    print(cv.groupby("k")["test_ll"].mean().round(3).to_string())
    print("\n--- K chosen per subject ---")
    print(tables["summary"]["best_k"].value_counts().sort_index().to_string())

    gains = tables["summary"]["cv_gain"]
    print(
        f"\nstates beat no states in {(gains > 0).sum()}/{len(gains)} subjects, "
        f"median gain {gains.median():+.3f} nats/sample"
    )

    print(f"\n--- behavioural coupling at {REPORT_LAG:.0f} s, second level over subjects ---")
    group = coupling.group_test(
        coup[coup.lag_s == REPORT_LAG], by=("outcome", "predictor"), value="slope"
    )
    contrast = coupling.group_test(
        coup[coup.lag_s == REPORT_LAG], by=("outcome", "predictor"), value="contrast"
    )
    group = pd.concat([group, contrast], ignore_index=True)
    group.to_csv(args.outdir / f"coupling_group_{args.tag}.csv", index=False)
    print(group[group.value == "slope"].round(4).to_string(index=False))

    # Full lag profile, since the right lag is an empirical question.
    lag_profile = coupling.group_test(coup, by=("outcome", "predictor", "lag_s"), value="slope")
    lag_profile.to_csv(args.outdir / f"coupling_lags_{args.tag}.csv", index=False)

    contrasts = profile_contrast(tables["profiles"])
    contrasts.to_csv(args.outdir / f"profile_contrast_{args.tag}.csv", index=False)

    hbo_hbr = tables["hbo_hbr"]
    print("\n--- HbO/HbR baseline correlation per state ---")
    print(
        f"negative in {(hbo_hbr.r_hbo_hbr < 0).sum()}/{len(hbo_hbr)} states, "
        f"median r = {hbo_hbr.r_hbo_hbr.median():.2f}"
    )
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
