import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from fnirs_glmhmm import behavior, io

    return behavior, io, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Explore the VTC median split

    Choose a subject to inspect their first run. VTC measures the absolute
    standardised reaction-time deviation; higher values indicate larger deviations.

    The table adds smoothed VTC and an `out_of_zone` label to the event data. In the
    plot, values above the smoothed series' median are labelled out of the zone.
    Compare the original and smoothed curves to see how the filter changes the
    duration of high- and low-VTC periods.

    For the HMM comparison and group analysis, open `zone_hmm_reproduction.py`.
    """)
    return


@app.cell
def _(io, mo):
    subs = io.subject_ids()
    sub = mo.ui.dropdown(subs, value=subs[0], label="subject")
    sub
    return (sub,)


@app.cell
def _(behavior, io, sub):
    events = behavior.zone_series(io.load_events(sub.value, run=1))
    events.head()
    return (events,)


@app.cell
def _(events, np, plt):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(events["VTC"], lw=0.5, color="0.7", label="VTC")
    ax.plot(events["VTC_smoothed"], lw=1.5, color="C0", label="smoothed")
    ax.axhline(np.median(events["VTC_smoothed"]), ls="--", color="C3", label="median split")
    ax.set(xlabel="trial", ylabel="VTC")
    ax.legend()
    fig
    return


if __name__ == "__main__":
    app.run()
