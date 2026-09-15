import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Zone HMM

    Refits the VTC results in `results/findings.md`, one subject at a time. Calls the same
    functions as `scripts/fit_zones.py` and `scripts/make_figdata.py`.

    Cells run on load, ~20 s. Buttons mark the slow ones.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    from fnirs_glmhmm import io, plotting
    from fnirs_glmhmm.behavior import signed_vtc, smooth_vtc, zone_labels
    from fnirs_glmhmm.zones import compare_to_median_split, fit_zone_hmm

    plotting.set_style()
    return (
        compare_to_median_split,
        fit_zone_hmm,
        io,
        mo,
        np,
        pd,
        plotting,
        plt,
        signed_vtc,
        smooth_vtc,
        wilcoxon,
        zone_labels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Controls

    `zones.prepare_vtc` is the only thing that touches VTC before EM.

    | transform | does | emissions |
    |---|---|---|
    | `log` | log1p, then z-score per run | Gaussian |
    | `zscore` | z-score per run | Gaussian |
    | `gamma` | nothing | Gamma |

    VTC is positive with a long right tail. Left raw, a Gaussian spends a state on the tail
    instead of on behaviour. Hence `log`.

    None of it is normal. Median skew across runs: RT +0.50, VTC +1.38, log1p(VTC) +0.73. The
    absolute value creates the skew, the log halves it. A Gaussian HMM needs the emissions
    normal within state, not the marginal, and the mixture absorbs the rest.

    States sort by emission mean either way, so state 0 is always the in-the-zone one.
    """)
    return


@app.cell
def _(io, mo):
    subs = io.subject_ids()
    sub = mo.ui.dropdown(subs, value=subs[0], label="subject")
    transform = mo.ui.radio(["log", "zscore", "gamma"], value="log", label="transform", inline=True)
    k = mo.ui.slider(2, 5, value=2, label="states")
    restarts = mo.ui.slider(2, 12, value=8, label="EM restarts")
    mo.hstack([sub, transform, k, restarts], justify="start", gap=2)
    return k, restarts, sub, transform


@app.cell
def _(io, sub):
    events = io.load_all_events(sub.value)
    vtc_runs = [e["VTC"].to_numpy() for e in events]
    f"{len(vtc_runs)} runs, {sum(len(v) for v in vtc_runs)} trials"
    return events, vtc_runs


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Why raw VTC

    The median split smooths first, 20-trial Gaussian kernel. Its long in/out blocks come from
    that kernel: raw VTC is lag-1 r ~ 0.5, in the noise by lag 10.

    Fit the smoothed series and the HMM fits the filter.
    """)
    return


@app.cell
def _(np):
    def acf(x, nlags=60):
        x = np.asarray(x, float) - np.mean(x)
        d = x @ x
        return np.array([x[: len(x) - lag] @ x[lag:] / d for lag in range(nlags + 1)])

    def iat(curve):
        """Integrated autocorrelation time, summed to the first non-positive lag."""
        stop = int(np.argmax(curve[1:] <= 0)) + 1
        return 1 + 2 * curve[1:stop].sum()

    return acf, iat


@app.cell
def _(acf, iat, np, plotting, plt, smooth_vtc, vtc_runs):
    _raw = np.mean([acf(v) for v in vtc_runs], axis=0)
    _sm = np.mean([acf(smooth_vtc(v)) for v in vtc_runs], axis=0)
    _floor = 2 / np.sqrt(np.mean([len(v) for v in vtc_runs]))

    _fig, _ax = plt.subplots(figsize=(6, 2.6))
    _ax.axhspan(-_floor, _floor, color=plotting.LIGHT, alpha=0.4, lw=0)
    _ax.plot(_raw, color=plotting.BLUE, label=f"raw, IAT {iat(_raw):.1f} trials")
    _ax.plot(_sm, color=plotting.GREEN, label=f"smoothed, IAT {iat(_sm):.1f} trials")
    _ax.axhline(0, color="0.5", lw=0.5)
    _ax.set(xlabel="lag (trials)", ylabel="autocorrelation")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The fit

    Runs share parameters. Concatenated, not batched, since dynamax has no masking and runs
    differ in length. Costs one false transition per run boundary.
    """)
    return


@app.cell
def _(fit_zone_hmm, k, restarts, transform, vtc_runs):
    fit = fit_zone_hmm(
        vtc_runs,
        num_states=k.value,
        transform=transform.value,
        num_restarts=restarts.value,
    )
    return (fit,)


@app.cell
def _(fit, np, pd):
    pd.DataFrame(
        {
            "emission mean": np.atleast_1d(fit.state_means),
            "dwell (trials)": fit.expected_dwell,
            "occupancy": fit.occupancy(),
        }
    ).rename_axis("state").round(3)
    return


@app.cell
def _(compare_to_median_split, fit, mo, np, vtc_runs):
    cmp = compare_to_median_split(fit, vtc_runs)
    mo.md(
        f"""
        Against the median split, collapsing the HMM to highest-VTC state vs the rest:
        agreement **{cmp["agreement"]:.3f}**, fraction out of the zone
        **{cmp["frac_out_hmm"]:.3f}** vs **{cmp["frac_out_median_split"]:.3f}**, switches per run
        **{np.mean(cmp["switches_hmm"]):.0f}** vs **{np.mean(cmp["switches_median_split"]):.0f}**.
        The split is pinned at 0.5 by construction; the HMM is not.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## One run

    VTC and its smoothed version, the HMM's `P(out)`, the split's hard label. Ticks are
    omissions.
    """)
    return


@app.cell
def _(mo, vtc_runs):
    run = mo.ui.dropdown(
        {f"run {i + 1}": i for i in range(len(vtc_runs))}, value="run 1", label="run"
    )
    run
    return (run,)


@app.cell
def _(events, fit, np, plotting, plt, run, smooth_vtc, vtc_runs, zone_labels):
    _v = vtc_runs[run.value]
    _p_out = fit.out_of_zone_prob()[run.value]
    _split, _ = zone_labels(smooth_vtc(_v))
    _omit = np.flatnonzero((events[run.value].response_code == -1).to_numpy())
    _t = np.arange(len(_v))

    _fig, _ax = plt.subplots(
        3, 1, figsize=(10, 4.5), sharex=True, height_ratios=[2.2, 1, 1], constrained_layout=True
    )
    _ax[0].plot(_t, _v, lw=0.4, color=plotting.LIGHT)
    _ax[0].plot(_t, smooth_vtc(_v), lw=1.4, color=plotting.BLUE)
    _ax[0].plot(_omit, np.full(len(_omit), _v.max()), "|", color=plotting.VERMILLION, ms=4)
    _ax[0].set_ylabel("VTC")

    _ax[1].fill_between(_t, _p_out, color=plotting.BLUE, alpha=0.6, lw=0)
    _ax[1].set(ylabel="P(out), HMM", ylim=(0, 1))

    _ax[2].fill_between(_t, _split, color=plotting.VERMILLION, alpha=0.6, lw=0, step="mid")
    _ax[2].set(ylabel="out, split", ylim=(0, 1), xlabel="trial")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Scoring

    Omissions, not commission errors: VTC is unrelated to commission errors here (0.498 vs
    0.499, p = 0.89), so every measure scores null on them.

    Lift = `P(omit | out) - P(omit | in)`, go trials, runs pooled.
    """)
    return


@app.cell
def _(np, pd, smooth_vtc, zone_labels):
    def omission_lift(events, out_mask):
        ev = pd.concat(events, ignore_index=True)
        go = (ev.trial_type == "city").to_numpy()
        omit = (ev.response_code == -1).to_numpy()
        out = np.asarray(out_mask, bool)
        a, b = go & out, go & ~out
        return (omit[a].mean() if a.any() else np.nan) - (omit[b].mean() if b.any() else np.nan)

    def hmm_runs(fit):
        return [s == fit.num_states - 1 for s in fit.states()]

    def split_runs(vtc_runs):
        return [zone_labels(smooth_vtc(v))[0].astype(bool) for v in vtc_runs]

    def hmm_mask(fit):
        return np.concatenate(hmm_runs(fit))

    def split_mask(vtc_runs):
        return np.concatenate(split_runs(vtc_runs))

    def switches(masks):
        """Mean transitions per run, so runs of different length weigh the same."""
        return float(np.mean([np.diff(m).astype(bool).sum() for m in masks]))

    return hmm_mask, hmm_runs, omission_lift, split_mask, split_runs, switches


@app.cell
def _(
    events,
    fit,
    hmm_mask,
    mo,
    omission_lift,
    split_mask,
    transform,
    vtc_runs,
):
    mo.md(f"""
    This subject: HMM (`{transform.value}`) **{omission_lift(events, hmm_mask(fit)):+.4f}**,
    median split **{omission_lift(events, split_mask(vtc_runs)):+.4f}**. One subject proves
    nothing; the group cell below is the comparison.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What the transform changes

    All three at k = 2. `gamma` means are on the raw VTC scale, so that row's means do not
    compare to the others.
    """)
    return


@app.cell
def _(mo):
    run_transforms = mo.ui.run_button(label="compare transforms (~45 s)")
    run_transforms
    return (run_transforms,)


@app.cell
def _(
    events,
    fit_zone_hmm,
    hmm_mask,
    hmm_runs,
    mo,
    omission_lift,
    pd,
    restarts,
    run_transforms,
    switches,
    vtc_runs,
):
    mo.stop(not run_transforms.value, mo.md("*not run*"))

    _rows = []
    for _tf in ("log", "zscore", "gamma"):
        _f = fit_zone_hmm(vtc_runs, num_states=2, transform=_tf, num_restarts=restarts.value)
        _rows.append(
            {
                "transform": _tf,
                "mean in": _f.state_means[0],
                "mean out": _f.state_means[1],
                "dwell in": _f.expected_dwell[0],
                "dwell out": _f.expected_dwell[1],
                "frac out": _f.occupancy()[1],
                "switches/run": switches(hmm_runs(_f)),
                "omission lift": omission_lift(events, hmm_mask(_f)),
            }
        )
    pd.DataFrame(_rows).set_index("transform").round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## State count

    BIC picks 4 over 2 in 16/21 subjects. Ignore it: omission forecast AUC is flat across the
    sweep (k=2 0.797, k=5 0.777, k=5 significantly worse). The extra states fit the skew, not
    extra regimes. Published numbers use k = 2.
    """)
    return


@app.cell
def _(mo):
    run_sweep = mo.ui.run_button(label="BIC sweep, k = 2-5 (~1 min)")
    run_sweep
    return (run_sweep,)


@app.cell
def _(fit_zone_hmm, mo, np, pd, restarts, run_sweep, transform, vtc_runs):
    mo.stop(not run_sweep.value, mo.md("*not run*"))

    _rows = []
    for _n in (2, 3, 4, 5):
        _f = fit_zone_hmm(
            vtc_runs, num_states=_n, transform=transform.value, num_restarts=restarts.value
        )
        _rows.append(
            {
                "states": _n,
                "BIC": _f.bic(),
                "log lik": _f.final_log_prob,
                "means": np.round(np.atleast_1d(_f.state_means), 2),
                "min dwell": _f.expected_dwell.min(),
            }
        )
    pd.DataFrame(_rows).set_index("states").round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## All subjects

    Per subject: HMM on raw VTC (transform above), HMM on smoothed VTC, median split. The
    smoothed fit uses `zscore`, matching `make_figdata.py` -- a 20-trial average is already
    near-Gaussian, skew 1.2 to 0.8.

    Published, with `log`:

    | measure | mean lift | p | subjects + |
    |---|---|---|---|
    | HMM, raw | +0.0233 | 0.0023 | 15/21 |
    | HMM, smoothed | +0.0107 | 0.0038 | 14/21 |
    | median split | +0.0101 | 0.0038 | 15/21 |

    Paired, raw HMM over split: +0.0132 (p = 0.0019). Part of that is timing, not better
    states: omissions sit in locally high VTC and the HMM switches faster.

    Two fits x 21 subjects, serial: ~10 min. `scripts/make_figdata.py` runs it on 12 cores.
    """)
    return


@app.cell
def _(mo):
    run_group = mo.ui.run_button(label="fit all subjects (~10 min)")
    run_group
    return (run_group,)


@app.cell
def _(
    fit_zone_hmm,
    hmm_mask,
    hmm_runs,
    io,
    mo,
    omission_lift,
    pd,
    restarts,
    run_group,
    smooth_vtc,
    split_mask,
    split_runs,
    switches,
    transform,
):
    mo.stop(not run_group.value, mo.md("*not run*"))

    import jax as _jax  # cell-local, so both group cells can clear its caches

    _rows = []
    for _sub in mo.status.progress_bar(io.subject_ids(), title="fitting", remove_on_exit=True):
        _ev = io.load_all_events(_sub)
        _v = [e["VTC"].to_numpy() for e in _ev]
        _raw = fit_zone_hmm(
            _v, num_states=2, transform=transform.value, num_restarts=restarts.value
        )
        _sm = fit_zone_hmm(
            [smooth_vtc(x) for x in _v],
            num_states=2,
            transform="zscore",
            num_restarts=restarts.value,
        )
        _rows.append(
            {
                "sub": _sub,
                f"HMM ({transform.value})": omission_lift(_ev, hmm_mask(_raw)),
                "HMM (smoothed)": omission_lift(_ev, hmm_mask(_sm)),
                "Median split": omission_lift(_ev, split_mask(_v)),
                "dwell_in": _raw.expected_dwell[0],
                "dwell_out": _raw.expected_dwell[1],
                "switches_hmm": switches(hmm_runs(_raw)),
                "switches_split": switches(split_runs(_v)),
            }
        )
        _jax.clear_caches()  # compilation caches grow across subjects and are never reused

    group = pd.DataFrame(_rows).set_index("sub")
    group.round(4)
    return (group,)


@app.cell
def _(group, pd, transform, wilcoxon):
    _measures = [f"HMM ({transform.value})", "HMM (smoothed)", "Median split"]
    _rows = []
    for _m in _measures:
        _x = group[_m].dropna()
        _rows.append(
            {
                "measure": _m,
                "mean lift": _x.mean(),
                "median": _x.median(),
                "Wilcoxon p": wilcoxon(_x).pvalue,
                "positive": f"{int((_x > 0).sum())}/{len(_x)}",
            }
        )

    _paired = (group[_measures[0]] - group["Median split"]).dropna()
    _rows.append(
        {
            "measure": "raw HMM - split, paired",
            "mean lift": _paired.mean(),
            "median": _paired.median(),
            "Wilcoxon p": wilcoxon(_paired).pvalue,
            "positive": f"{int((_paired > 0).sum())}/{len(_paired)}",
        }
    )
    pd.DataFrame(_rows).set_index("measure").round(4)
    return


@app.cell
def _(group, mo, np, transform):
    mo.md(f"""
    Medians: dwell **{np.median(group.dwell_in):.1f}** trials in the zone,
    **{np.median(group.dwell_out):.1f}** out, **{np.median(group.switches_hmm):.0f}** switches
    per run against the split's **{np.median(group.switches_split):.0f}**. Matches VTC's
    autocorrelation time, so the split's long blocks are the kernel. Transform:
    `{transform.value}`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Dropping the absolute value

    VTC is |z(RT)|. A trial 1.5 sd fast and one 1.5 sd slow are the same number, so they land
    in the same state. Out of the zone means variable, not slow.

    `behavior.signed_vtc` puts the sign back. `abs()` of it matches the `VTC` column to 4e-4
    on all 62 runs, the column's own rounding. Getting that match pinned two upstream
    conventions:

    - mean and sd include non-responses at RT = 0, so the centre sits below the
      responded-trial mean and the sd is wide;
    - non-response trials copy the previous responded trial forward. No interpolation. An
      omission inherits the deviation of the trial before it.

    Fit it with `transform="none"`: already standardised, and re-centring would move the
    fast/slow boundary off zero.
    """)
    return


@app.cell
def _(events, mo, np, signed_vtc):
    signed_runs = [signed_vtc(e) for e in events]
    _err = max(
        np.abs(np.abs(v) - e["VTC"].to_numpy()).max()
        for v, e in zip(signed_runs, events, strict=True)
    )
    _fast = np.mean(np.concatenate(signed_runs) < 0)
    mo.md(f"""
    Largest `|abs(signed) - VTC|` here: **{_err:.1e}**. **{_fast:.1%}** of trials are on the
    fast side.
    """)
    return (signed_runs,)


@app.cell
def _(mo):
    k_signed = mo.ui.slider(2, 4, value=3, label="states, signed VTC")
    k_signed
    return (k_signed,)


@app.cell
def _(fit_zone_hmm, k_signed, restarts, signed_runs):
    fit_signed = fit_zone_hmm(
        signed_runs, num_states=k_signed.value, transform="none", num_restarts=restarts.value
    )
    return (fit_signed,)


@app.cell
def _(events, fit_signed, np, pd):
    def state_rates(fit, events):
        """Occupancy, dwell and omission rate per state, states ascending in RT deviation."""
        ev = pd.concat(events, ignore_index=True)
        go = (ev.trial_type == "city").to_numpy()
        omit = (ev.response_code == -1).to_numpy()
        st = np.concatenate(fit.states())
        rates = []
        for s in range(fit.num_states):
            m = go & (st == s)
            rates.append(omit[m].mean() if m.any() else np.nan)
        return pd.DataFrame(
            {
                "mean (sd of RT)": fit.state_means,
                "occupancy": fit.occupancy(),
                "dwell (trials)": fit.expected_dwell,
                "P(omit | go)": rates,
            }
        ).rename_axis("state")

    state_rates(fit_signed, events).round(3)
    return (state_rates,)


@app.cell
def _(fit_signed, np, plotting, plt, run, signed_runs):
    _v = signed_runs[run.value]
    _st = fit_signed.states()[run.value]
    _t = np.arange(len(_v))
    _shade = plt.cm.coolwarm(np.linspace(0.15, 0.85, fit_signed.num_states))
    _lim = max(4, np.abs(_v).max())

    _fig, _ax = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    for _s in range(fit_signed.num_states):
        _ax.fill_between(
            _t, -_lim, _lim, where=_st == _s, color=_shade[_s], alpha=0.35, lw=0, step="mid"
        )
    _ax.plot(_t, _v, lw=0.6, color=plotting.GREY)
    _ax.axhline(0, color="black", lw=0.6)
    _ax.set(
        xlabel="trial",
        ylabel="signed VTC (sd)",
        ylim=(-_lim, _lim),
        title="below the line is fast, above is slow; shading is the state",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Which tail

    Slowest state minus the rest, fastest state minus the rest, folded k = 2 fit for
    reference.
    """)
    return


@app.cell
def _(mo):
    run_signed_group = mo.ui.run_button(label="fit all subjects, signed (~10 min)")
    run_signed_group
    return (run_signed_group,)


@app.cell
def _(
    fit_zone_hmm,
    hmm_mask,
    io,
    k_signed,
    mo,
    np,
    omission_lift,
    pd,
    restarts,
    run_signed_group,
    signed_vtc,
    state_rates,
    transform,
):
    mo.stop(not run_signed_group.value, mo.md("*not run*"))

    import jax as _jax  # cell-local, so both group cells can clear its caches

    _rows = []
    for _sub in mo.status.progress_bar(io.subject_ids(), title="fitting", remove_on_exit=True):
        _ev = io.load_all_events(_sub)
        _sg = fit_zone_hmm(
            [signed_vtc(e) for e in _ev],
            num_states=k_signed.value,
            transform="none",
            num_restarts=restarts.value,
        )
        _folded = fit_zone_hmm(
            [e["VTC"].to_numpy() for e in _ev],
            num_states=2,
            transform=transform.value,
            num_restarts=restarts.value,
        )
        _st = np.concatenate(_sg.states())
        _rates = state_rates(_sg, _ev)["P(omit | go)"].to_numpy()
        _rows.append(
            {
                "sub": _sub,
                "P(omit) fastest": _rates[0],
                "P(omit) slowest": _rates[-1],
                "slowest - rest": omission_lift(_ev, _st == _sg.num_states - 1),
                "fastest - rest": omission_lift(_ev, _st == 0),
                "folded abs(VTC)": omission_lift(_ev, hmm_mask(_folded)),
            }
        )
        _jax.clear_caches()

    signed_group = pd.DataFrame(_rows).set_index("sub")
    signed_group.round(4)
    return (signed_group,)


@app.cell
def _(pd, signed_group, wilcoxon):
    _rows = []
    for _c in ("slowest - rest", "fastest - rest", "folded abs(VTC)"):
        _x = signed_group[_c].dropna()
        _rows.append(
            {
                "contrast": _c,
                "mean lift": _x.mean(),
                "median": _x.median(),
                "Wilcoxon p": wilcoxon(_x).pvalue,
                "positive": f"{int((_x > 0).sum())}/{len(_x)}",
            }
        )
    pd.DataFrame(_rows).set_index("contrast").round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    k = 3 gives fast (-0.96 sd, occupancy 0.31, dwell 5.3), on pace (+0.11, 0.50, 5.7) and
    slow (+1.31, 0.16, 3.3).

    | contrast | mean lift | p | subjects + |
    |---|---|---|---|
    | slowest - rest | +0.0390 | 0.0005 | 15/21 |
    | fastest - rest | -0.0060 | 0.0052 | 1/21 |
    | folded, abs(VTC) | +0.0233 | 0.0023 | 15/21 |

    The fast tail is not a lapse state: it omits no more than on pace (+0.003, p = 0.80). The
    slow tail does (+0.040, p = 0.0006). Folding them costs +0.0157 of lift (paired,
    p = 0.013, 12/21).

    Two caveats. The slow state holds 0.16 of trials against ~0.35 folded, so some of the gain
    is selectivity. And the forward-fill makes a slow drift into an omission partly
    definitional; the lag-exclusion controls in `findings.md` ran on the folded measure only.
    """)
    return


if __name__ == "__main__":
    app.run()
