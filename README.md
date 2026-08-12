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
  behavior.py  VTC smoothing and in/out-of-zone labels
  model.py     dynamax GLM-HMM wrappers
notebooks/     marimo notebooks (plain .py, diffable)
tests/
```

Notebooks are marimo, so they're regular Python files and review like code:

```bash
uv run marimo edit notebooks/explore_vtc.py
uv run marimo run notebooks/explore_vtc.py   # read-only app view
```

On the cluster add `--headless --port 2718` and forward the port over ssh.

Dataset paths default to the copy on `/projectnb/nphfnirs`; override with
`GRADCPT_ROOT` if you're working off a local copy.

## Usage

```python
from fnirs_glmhmm import behavior, io, model

runs = io.load_parcel_ts("sub-01", variant="ols")  # list of (chromophore, parcel, time)
events = behavior.zone_series(io.load_events("sub-01", run=1))

fit = model.fit_glm_hmm(
    emissions=events["out_of_zone"].to_numpy(),
    inputs=design,  # (n_trials, n_features)
    num_states=3,
    kind="logistic",
)
states = fit.most_likely_states(events["out_of_zone"].to_numpy(), inputs=design)
```

`model.state_sweep` fits a range of state counts if you want to look at the
log-likelihood curve before committing to one.

## Notes

- Two preprocessing variants per subject: `ols` (TDDR + 0.5 Hz lowpass before
  parcellation) and `ar_irls` (no preprocessing before parcellation).
- The medial wall and scalp parcels are dropped on load; see
  `config.NON_CORTICAL_PARCELS`.
- dynamax needs equal-length sessions to fit a batch. `model.stack_sessions`
  truncates ragged runs and warns; fit separately if that matters.
- Everything runs on CPU JAX. Add a CUDA jax build if a GPU node is available.
