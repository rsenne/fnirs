import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Comparing HMM zones with the median split

    We want to know how much the zone result depends on smoothing and the median
    threshold. We fit an HMM to unsmoothed VTC, compare its labels with the usual
    median split, and check how both relate to omissions.

    The analysis starts with one subject so we can inspect the fit before running
    the group comparison. We then vary the emission transform and state count.
    Finally, we fit signed VTC to ask whether fast and slow deviations have different
    associations with omissions.

    The initial fits run on load. Buttons start the longer comparisons; their timing
    estimates depend on your machine.
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
    ## 1. Choose the fit

    We start with two states to keep the comparison close to the binary median split.
    All runs from the selected subject contribute to the fit.

    The default log transform reduces the influence of VTC's right tail. We also
    try standardisation alone and Gamma emissions to check whether the result depends
    on that treatment of the distribution.

    | Transform | Preparation within each run | Emissions |
    |---|---|---|
    | `log` | log1p, then z-score | Gaussian |
    | `zscore` | z-score | Gaussian |
    | `gamma` | Original scale, with a small offset if zeros are present | Gamma |

    We order states by emission mean and call the highest-VTC state "out of the zone."
    For fits with more than two states, the binary comparison combines all other states.

    The restarts control repeats the fit from different initialisations. We keep the
    fit with the highest final log likelihood to reduce sensitivity to the starting
    point.
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
    ## 2. Check the effect of smoothing

    The median split uses a 20-trial Gaussian filter, applied forwards and backwards.
    Before comparing labels, we check how much persistence this adds to the series.

    The autocorrelation curves below are averaged across the selected subject's runs.
    The legend reports integrated autocorrelation time, summed up to the first
    non-positive lag. The shaded band is a rough reference of +/- 2 / sqrt(N), using
    the average run length.

    We use unsmoothed VTC for the main HMM so the filter does not set the persistence
    the model sees. Later, we also fit smoothed VTC to assess how much of the difference
    between methods comes from this choice.
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
    ## 3. Fit all runs from this subject

    Sharing parameters across runs gives us one set of states for each subject.
    The implementation concatenates runs during fitting to accommodate their different
    lengths. This adds an artificial transition at each run boundary; decoding is
    performed separately for each run.

    Check the state means alongside occupancy and dwell. A state that captures only
    a few extreme trials gives a different segmentation from one occupied throughout
    much of the run. Dwell comes from the fitted self-transition probability,
    `1 / (1 - p_stay)`; occupancy comes from the Viterbi path.
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
        Treating the highest-VTC state as "out," the HMM agrees with the median
        split on **{cmp["agreement"]:.1%}** of trials, averaged across runs.

        The mean fraction out of the zone is **{cmp["frac_out_hmm"]:.3f}** for the
        HMM and **{cmp["frac_out_median_split"]:.3f}** for the median split.
        They average **{np.mean(cmp["switches_hmm"]):.0f}** and
        **{np.mean(cmp["switches_median_split"]):.0f}** switches per run, respectively.

        The median split labels half of each run out of the zone; ties
        at the median are labelled in. The HMM's fraction depends on the fit.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Inspect where the labels differ

    The top panel overlays original and smoothed VTC, with red ticks for omissions.
    The next two panels show the HMM's `P(out)` and the median split's binary label.

    Look at both the location and duration of out-of-zone periods. Differences in
    omission rates may reflect how quickly the labels track local changes in VTC.

    The HMM probabilities use the full run, including later observations. This plot
    describes the recorded sequence; it is not a prospective prediction.
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
    ## 5. Compare omission rates

    We use omissions to assess whether the segmentations identify periods with
    different error rates. The commission-error comparison is available separately
    in `scripts/fit_zones.py`.

    Within each subject, we pool runs and calculate:

    `omission lift = P(omission | out, city) - P(omission | in, city)`

    Pooling gives us more observations for each conditional rate. For the HMM, "out"
    means the highest-VTC state in the Viterbi path. A lift of 0.02 is a two-percentage-
    point difference in omission rates.

    This is a same-run association. VTC on non-response trials is filled from responded
    trials, so the score also depends on that construction. We return to this when
    interpreting the signed-VTC fit.
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
    For this subject, omission lift is
    **{omission_lift(events, hmm_mask(fit)):+.4f}** for the HMM (`{transform.value}`)
    and **{omission_lift(events, split_mask(vtc_runs)):+.4f}** for the median split.
    The group comparison below checks how these differences vary across subjects.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Check sensitivity to the transform

    Refit the selected subject with all three transforms, holding the state count at
    two and using the selected number of restarts.

    Compare omission lift together with occupancy, dwell, and switching. Similar
    results across transforms would suggest that the segmentation is not especially
    sensitive to our treatment of the VTC distribution.

    Emission means are on different scales across rows, so compare their ordering
    within a fit rather than their numerical values between transforms.
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
    ## 7. Check what extra states capture

    The sweep fits two through five states with the selected transform. BIC asks
    whether the improvement in likelihood offsets the extra parameters.

    Inspect the state means and minimum dwell alongside BIC. Extra states may describe
    more detail in the VTC distribution without identifying additional attentional
    conditions. A lower BIC does not resolve that interpretation by itself.

    We keep two states for the group comparison so every subject contributes the
    same binary contrast. The sweep does not change that setting.
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
    ## 8. Compare subjects

    For each subject, we calculate omission lift for:

    1. An HMM on unsmoothed VTC with the selected transform.
    2. An HMM on smoothed VTC with `zscore`.
    3. The median split of smoothed VTC.

    Both HMMs use two states and the selected number of restarts. The smoothed fit
    matches `scripts/make_figdata.py`. It provides a comparison with the median split
    on the same input series. With the default settings, the two HMMs differ in both
    smoothing and transform, so their contrast does not isolate smoothing alone.

    We calculate effects within subject before summarising across subjects. The final
    row tests the paired difference in lift between the unsmoothed HMM and median split.
    The Wilcoxon p-values are unadjusted for the comparisons shown.

    Read lift alongside occupancy and switching: a larger contrast may reflect a
    smaller or more precisely timed set of out-of-zone trials.

    This cell runs subjects serially and may take about ten minutes. The batch script
    `scripts/make_figdata.py` runs the default comparison with 12 workers.
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
    Across subjects, the median expected HMM dwell is
    **{np.median(group.dwell_in):.1f}** trials in the zone and
    **{np.median(group.dwell_out):.1f}** out of the zone, using `{transform.value}`.
    The median of subjects' mean switch counts is
    **{np.median(group.switches_hmm):.0f}** per run for the HMM and
    **{np.median(group.switches_split):.0f}** for the median split.
    Compare these with the autocorrelation plot to assess the methods' timescales.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. Separate fast and slow deviations

    The unsigned fit combines fast and slow deviations. We now keep the sign to ask
    whether they have different associations with omissions.

    `behavior.signed_vtc` reconstructs the series using the dataset's conventions:
    RT = 0 non-responses enter the mean and standard deviation, then take the most
    recent responded trial's deviation. Leading non-responses use the first response.
    The check below compares the reconstructed absolute values with the supplied VTC.

    We use `transform="none"` to preserve this scale and its zero point. The default
    three-state fit allows low, intermediate, and high signed deviations to separate.
    Their means determine the interpretation; the model does not impose fast,
    on-pace, and slow categories.

    The table reports omission rates by state. The plot uses the run selected above.
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
    The largest absolute difference between reconstructed and supplied VTC is
    **{_err:.1e}** for this subject. Signed VTC is below zero on **{_fast:.1%}** of
    trials, including filled values on trials without a response.
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
        title="Signed reaction-time deviations and fitted states",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Compare each tail with the remaining states

    For each subject, compare the highest-mean signed state with all remaining states,
    then repeat for the lowest-mean state. We also refit unsigned VTC with two states
    as a reference.

    The signed fits use the state count selected above; the unsigned fits use the
    transform selected at the start. All use the selected number of restarts.
    This checks whether combining the two tails obscures different omission rates.
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
    ## 10. Interpret the signed comparison

    Check whether the slowest and fastest states have similar omission contrasts.
    If they differ, a single high-VTC category combines periods with different
    behavioural associations.

    Compare occupancy as well as lift. A smaller state can select a narrower set of
    high-error periods, so a larger lift alone does not establish that the signed
    model is a better description.

    An omission inherits a responded trial's signed deviation. An association with
    the slowest state therefore describes the surrounding response pattern, not the
    reaction time of the missed response. A predictive analysis would need to restrict
    the information available before each trial.
    """)
    return


if __name__ == "__main__":
    app.run()
