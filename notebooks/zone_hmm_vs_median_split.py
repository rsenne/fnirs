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


@app.cell
def _(np, state_sweep, vtc_runs):
    sweep = state_sweep(vtc_runs, state_range=(2, 3, 4), num_restarts=6)
    for n, f in sweep.items():
        print(f"{n} states  BIC {f.bic():9.1f}  means {np.round(f.state_means, 2)}")
    return
