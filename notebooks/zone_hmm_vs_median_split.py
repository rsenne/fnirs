import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from fnirs_glmhmm import io
    from fnirs_glmhmm.behavior import smooth_vtc, zone_labels
    from fnirs_glmhmm.zones import compare_to_median_split, fit_zone_hmm, state_sweep

    return (
        compare_to_median_split,
        fit_zone_hmm,
        io,
        mo,
        np,
        plt,
        smooth_vtc,
        state_sweep,
        zone_labels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compare HMM zones with the median split

    Choose a subject and a state count to fit an HMM to all of their VTC runs.
    VTC is the absolute standardised reaction-time deviation. The default fit uses
    log1p followed by standardisation within each run.

    States are ordered by their emission means. The binary comparison calls the
    highest-VTC state "out of the zone" and combines the others as "in." The median
    split labels trials above the median of smoothed VTC as out.

    This is a compact exploratory notebook. For a guided analysis with omission-rate
    comparisons, use `zone_hmm_reproduction.py`. All fits here run automatically.
    """)
    return


@app.cell
def _(io, mo):
    subs = io.subject_ids()
    sub = mo.ui.dropdown(subs, value=subs[0], label="subject")
    k = mo.ui.slider(2, 5, value=2, label="states")
    mo.hstack([sub, k], justify="start")
    return k, sub


@app.cell
def _(io, sub):
    vtc_runs = [e["VTC"].to_numpy() for e in io.load_all_events(sub.value)]
    return (vtc_runs,)


@app.cell
def _(fit_zone_hmm, k, vtc_runs):
    fit = fit_zone_hmm(vtc_runs, num_states=k.value, num_restarts=10)
    return (fit,)


@app.cell
def _(compare_to_median_split, fit, np, vtc_runs):
    cmp = compare_to_median_split(fit, vtc_runs)
    print(f"state means (log VTC, z): {np.round(fit.state_means, 2)}")
    print(f"expected dwell (trials):  {np.round(fit.expected_dwell, 1)}")
    print(f"occupancy:                {np.round(fit.occupancy(), 3)}")
    print(f"agreement with split:     {cmp['agreement']:.3f}")
    print(f"frac out: hmm {cmp['frac_out_hmm']:.3f} vs split {cmp['frac_out_median_split']:.3f}")
    print(f"switches: hmm {cmp['switches_hmm']} vs split {cmp['switches_median_split']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Read the fit and compare the first run

    The output above lists emission means, expected dwell in trials, occupancy, and
    agreement with the median split. Occupancy is the fraction of trials assigned to
    each state; dwell comes from the fitted transition probabilities.

    Below, the first panel overlays original and smoothed VTC. The next two panels
    show the HMM's probability of the highest-VTC state and the median split's binary
    label. HMM probabilities use the full run, including later observations.
    """)
    return


@app.cell
def _(fit, np, plt, smooth_vtc, vtc_runs, zone_labels):
    # Run 1: what each method calls out of the zone.
    vtc = vtc_runs[0]
    p_out = fit.out_of_zone_prob()[0]
    split_out, _ = zone_labels(smooth_vtc(vtc))

    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(vtc, lw=0.5, color="0.6")
    axes[0].plot(smooth_vtc(vtc), lw=1.5, color="C0")
    axes[0].set_ylabel("VTC")

    axes[1].fill_between(np.arange(len(p_out)), p_out, color="C3", alpha=0.6, lw=0)
    axes[1].set_ylabel("P(out) HMM")
    axes[1].set_ylim(0, 1)

    axes[2].fill_between(
        np.arange(len(split_out)), split_out, color="C7", alpha=0.8, lw=0, step="mid"
    )
    axes[2].set_ylabel("out, split")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("trial")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare state counts

    This cell fits two, three, and four states. Lower BIC indicates a better tradeoff
    between likelihood and parameter count under this criterion. Inspect the emission
    means too: additional states may describe details of the VTC distribution without
    necessarily identifying distinct attentional conditions.
    """)
    return


@app.cell
def _(np, state_sweep, vtc_runs):
    sweep = state_sweep(vtc_runs, state_range=(2, 3, 4), num_restarts=6)
    for n, f in sweep.items():
        print(f"{n} states  BIC {f.bic():9.1f}  means {np.round(f.state_means, 2)}")
    return
