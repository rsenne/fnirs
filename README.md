# fnirs-glmhmm

GLM-HMM analysis of the gradCPT fNIRS dataset (`gradCPT_NN24`, n = 21).

## Setup

```bash
uv sync
```

That creates `.venv` with everything pinned in `uv.lock`. Run things with
`uv run python ...` or `uv run pytest`, or activate the venv directly.

## Layout

```
src/fnirs_glmhmm/
  config.py    dataset paths and constants
  io.py        parcel timeseries, event tables, eyetracking
  behavior.py  VTC smoothing and the classic median-split zone labels
  zones.py     HMM-derived zones, the replacement for the median split
  model.py     dynamax GLM-HMM wrappers
  plotting.py  shared figure style and palette
scripts/       batch jobs that write to results/
  fit_zones.py     per-subject zone HMMs and the BIC sweep
  make_figdata.py  tidy CSVs for the figures, run before scripts/figures/
  figures/         one script per figure, reads results/figdata/
notebooks/     marimo notebooks (plain .py, diffable)
tests/
```

`scripts/fit_zones.py` fits every subject and scores the HMM zones against the
median split on commission errors. One process per subject, sized off `NSLOTS`.

Results and analysis notes live in `results/findings.md`, gitignored along with
the rest of `results/` and `figures/`.

Notebooks are marimo, so they're regular Python files:

```bash
uv run marimo edit notebooks/explore_vtc.py
uv run marimo run notebooks/explore_vtc.py   # read-only app view
```

On the cluster add `--headless --port 2718` and forward the port over ssh.

Dataset paths default to the copy on `/projectnb/nphfnirs`; override with
`GRADCPT_ROOT` if you're working off a local copy.

## Zones from an HMM

The in/out-of-the-zone measure smooths the VTC with a fixed Gaussian kernel
and splits at the median. The kernel width sets the switching timescale by hand,
and the median forces exactly half of every run out of the zone. `zones.py` fits an
HMM to the VTC instead:

```python
from fnirs_glmhmm import io
from fnirs_glmhmm.zones import compare_to_median_split, fit_zone_hmm

vtc = [e["VTC"].to_numpy() for e in io.load_all_events("sub-629")]

fit = fit_zone_hmm(vtc, num_states=2)  # runs share parameters
fit.states()  # Viterbi path per run, 0 = most in the zone
fit.out_of_zone_prob()  # graded P(out) per trial; prefer this to the hard label
fit.expected_dwell  # trials per state, straight off the transition matrix

compare_to_median_split(fit, vtc)
```

States are always relabelled by ascending VTC, so state 0 means the same thing for
every subject. `zones.state_sweep` fits a range of state counts and each fit
exposes `.bic()`.

VTC is a non-negative skewed deviation score, so `transform="log"` (log1p +
z-score, Gaussian emissions) is the default. `transform="gamma"` fits a GammaHMM on
the raw values if you'd rather not transform.

## GLM-HMM

```python
from fnirs_glmhmm import model

fit = model.fit_glm_hmm(
    emissions=y,  # e.g. zone label per trial
    inputs=design,  # (n_trials, n_features)
    num_states=3,
    kind="logistic",
)
states = fit.most_likely_states(y, inputs=design)
```