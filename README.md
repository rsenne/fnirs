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
  neural.py    network features, task design and nuisance regression for the fNIRS fit
  coupling.py  per-trial coupling between neural states and behaviour
  model.py     dynamax GLM-HMM wrappers
  plotting.py  shared figure style and palette
scripts/       batch jobs that write to results/
  fit_zones.py       per-subject zone HMMs and the BIC sweep
  fit_neural_hmm.py  per-subject neural GLM-HMMs, K by cross-validation
  neural_nulls.py    AR surrogate and circular-shift nulls for those fits
  make_figdata.py    tidy CSVs for the figures, run before scripts/figures/
  figures/           one script per figure, reads results/figdata/ or results/neural/
notebooks/     marimo notebooks (plain .py, diffable)
tests/
```

`scripts/fit_zones.py` fits every subject and scores the HMM zones against the
median split on commission errors. One process per subject, sized off `NSLOTS`.

Results and analysis notes live in `results/findings.md`, gitignored along with
the rest of `results/` and `figures/`.

Notebooks are marimo, so they're regular Python files:

```bash
uv run marimo edit notebooks/zone_hmm_reproduction.py
uv run marimo run notebooks/zone_hmm_reproduction.py   # read-only app view
```

Start with `zone_hmm_reproduction.py`. It refits the whole VTC result, with the transform
and the signed VTC as controls. The other two notebooks are scratch.

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

### The absolute value

VTC is |z(RT)|: a fast trial and a slow trial of the same size are one number, in one
state. Out of the zone means variable, not slow.

`behavior.signed_vtc` puts the sign back. `abs()` of it reproduces the `VTC` column to
4e-4 on all 62 runs, which pinned down two upstream conventions: mean and sd include
non-responses at RT = 0, and a non-response trial copies the previous responded trial
forward rather than being interpolated, so an omission inherits the deviation before it.

```python
from fnirs_glmhmm.behavior import signed_vtc

signed = [signed_vtc(e) for e in io.load_all_events("sub-629")]
fit = fit_zone_hmm(signed, num_states=3, transform="none")  # fast / on pace / slow
```

`transform="none"` passes the series through; it is already standardised, and re-centring
would move the fast/slow boundary off zero.

Split that way, omissions are slow-only: slowest state +0.039 (p = 0.0005) against +0.023
folded, fastest state -0.006. Details in `results/findings.md`.

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

## Neural GLM-HMM

The model that matters is fit to the fNIRS timeseries, not to behaviour:

    z_n ~ Markov(K states)
    y_n | z_n = k  ~  N(A_k x_n + b_k, Sigma_k)

`y_n` is network activity, `b_k` is a state's baseline network configuration, `A_k`
is its task response gain, and `x_n` is the HRF-convolved task. Behaviour never
enters the fit, so scoring states against behaviour afterwards is out of sample.

```python
from fnirs_glmhmm import coupling, neural

runs = neural.prepare_subject("sub-629")  # one RunData per usable run
y, x, meta = neural.stack(runs)  # (n_runs, T, 14) and (n_runs, T, 3)
```

Features are seven Yeo networks x two chromophores. Limbic is dropped (the median
subject has one covered parcel there, eight have none) and TempPar is kept separate
rather than folded into Default, which is what lands the count on 14. Per subject,
parcels are masked at HbO sd > 1e-7, averaged within network, bandpassed
0.01-0.2 Hz, resampled to 1 Hz, residualised on the global signal and a DCT drift
basis, then z-scored per run.

Inputs are `commission`, `correct_rejection` and `rt_mod`. The plan's `nogo`
regressor is absent on purpose: every mountain trial is either a commission error or
a correct rejection, so it is exactly the sum of the first two and the design would
be singular. Recover it as their sum.

**Models are fit per subject.** Nothing is pooled: each subject picks their own K by
leaving out one of their own runs, and the group claim is a test across per-subject
effects. States are comparable across subjects only through their DMN baseline, so
the highest- and lowest-`Default_HbO` states are what get carried up.

Covariances are diagonal (`--cov diag`, the default). At D = 14 a full covariance is
105 parameters per state against ~700 training samples with lag-1 autocorrelation
near 0.95, and it does not cross-validate: held-out likelihood comes out several
times worse than the one-state baseline. The K sweep includes that one-state
baseline as k = 1, so "no state structure" can win outright.

```bash
uv run python scripts/fit_neural_hmm.py    # per-subject fits, K sweep, coupling
uv run python scripts/neural_nulls.py      # AR surrogate and circular-shift nulls
uv run python scripts/figures/fig6_neural_k_selection.py
```

Both write to `results/neural/`. The nulls script reads the fits, so run it second.