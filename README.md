# fnirs-glmhmm

GLM-HMM analysis of the gradCPT fNIRS dataset (`gradCPT_NN24`, 21 subjects).
The repository includes an HMM analysis of reaction-time variability and a
GLM-HMM analysis of network-level fNIRS activity.

## Setup

```bash
uv sync
```

This installs the dependencies pinned in `uv.lock` into `.venv`.
Use `uv run` to run scripts, notebooks, or tests in that environment.

The dataset is stored separately. Paths default to
`/projectnb/nphfnirs/s/datasets/gradCPT_NN24`; set `GRADCPT_ROOT` to the dataset
directory if you have a local copy. Subject discovery checks for the parcel
timeseries files, so the notebook needs the expected derivatives directory as
well as the event tables.

## Start with the notebook

```bash
uv run marimo edit notebooks/zone_hmm_reproduction.py
uv run marimo run notebooks/zone_hmm_reproduction.py
```

The first command opens the editor; the second opens the notebook as an app.
The notebook walks through fitting one subject, comparing HMM zones with a
smoothed median split, checking modelling choices, and comparing omission rates
across subjects. It then uses signed reaction-time deviations to separate fast
and slow responses. Buttons start the longer fits.

The notebooks are plain Python files managed by marimo.
`explore_vtc.py` plots the median split for one run;
`zone_hmm_vs_median_split.py` provides a shorter comparison with the HMM.

On the cluster, add `--headless --port 2718` to the marimo command and forward
that port over SSH.

## Repository layout

```text
src/fnirs_glmhmm/
  config.py    dataset paths and analysis constants
  io.py        parcel timeseries, event tables, and eyetracking
  behavior.py  VTC smoothing, signed VTC, and median-split labels
  zones.py     HMM fits to VTC and other per-trial features
  neural.py    network features, task design, and nuisance regression
  coupling.py  associations between neural states and behaviour
  model.py     dynamax GLM-HMM wrappers
  plotting.py  shared figure style and palette
scripts/
  fit_zones.py       zone HMMs, commission-error comparison, and BIC sweep
  make_figdata.py    omission comparisons and descriptive CSVs
  fit_neural_hmm.py  neural GLM-HMMs and cross-validation over state counts
  neural_nulls.py    surrogate and circular-shift comparisons
notebooks/           marimo notebooks
tests/
```

Analysis outputs go to `results/`, which is gitignored. Local analysis notes may
also be kept in `results/findings.md`; that file is not included in this checkout.

## Zones from reaction-time variability

The variance time course (VTC) is the absolute standardised reaction-time
deviation, with filled values on trials without a response. Large VTC can reflect
either a fast or a slow response.

The median-split measure smooths VTC with a Gaussian filter and labels values
above the run's median as out of the zone. The filter affects the duration of
the resulting blocks, and the split assigns roughly half the trials to each zone.

The HMM estimates emission distributions and transition probabilities from
unsmoothed VTC. These describe the values within each state and how persistent
the states are.

```python
from fnirs_glmhmm import io
from fnirs_glmhmm.zones import compare_to_median_split, fit_zone_hmm

vtc = [e["VTC"].to_numpy() for e in io.load_all_events("sub-629")]
fit = fit_zone_hmm(vtc, num_states=2)

fit.states()            # most likely state path for each run
fit.out_of_zone_prob()   # probability of the highest-VTC state at each trial
fit.expected_dwell      # expected consecutive trials per state

compare_to_median_split(fit, vtc)
```

States are ordered by ascending emission mean. For unsigned VTC, state 0 has the
smallest deviations. With more than two states, the binary comparison treats the
highest-VTC state as out of the zone and combines the rest.

The default `transform="log"` applies log1p and standardises each run before
fitting Gaussian emissions. `transform="zscore"` standardises without the log;
`transform="gamma"` fits Gamma emissions on the VTC scale, adding a small offset
if zeros are present. `zones.state_sweep` fits several state counts, and each
fit exposes `.bic()` for comparison within the same transform.

Runs share fitted parameters. They are concatenated during fitting, introducing
an artificial transition at each boundary, then decoded separately. State
probabilities use the full run, so they describe the recorded data rather than
provide a forecast.

### Keep the direction of the deviation

`behavior.signed_vtc` reconstructs the standardised deviations before the
absolute value. Negative values are faster and positive values slower relative
to the mean used for standardisation.

```python
from fnirs_glmhmm.behavior import signed_vtc

signed = [signed_vtc(e) for e in io.load_all_events("sub-629")]
fit = fit_zone_hmm(signed, num_states=3, transform="none")
```

This reconstruction includes RT = 0 non-responses in the mean and standard
deviation. It then fills non-response trials with the most recent responded
trial's deviation, using the first response for any leading non-responses.
The notebook checks the reconstructed absolute values against the supplied VTC.

`transform="none"` preserves the signed scale and its zero point. States are
still ordered by mean, so state 0 now has the lowest signed deviation. When
comparing omission rates, remember that an omission's VTC comes from another
trial's response.

### Batch zone analyses

```bash
uv run python scripts/fit_zones.py
uv run python scripts/make_figdata.py
```

The first script writes `results/zone_hmm_vs_split.csv`, including a BIC sweep
and commission-error rates. It uses subject-level worker processes, with the
default worker count based on `NSLOTS`.

The second uses 12 workers and writes omission comparisons, autocorrelations,
dwell times, and example traces to `results/figdata/`. If the first script's
output is present, it also copies the BIC columns into `bic.csv`.

## GLM-HMM interface

A GLM-HMM lets the relationship between inputs and observations vary by state.
For example, the logistic model below uses a design matrix to model a binary
observation such as a zone label.

```python
from fnirs_glmhmm import model

fit = model.fit_glm_hmm(
    emissions=y,       # binary observations
    inputs=design,     # (n_trials, n_features)
    num_states=3,
    kind="logistic",
)
states = fit.most_likely_states(y, inputs=design)
```

## Neural GLM-HMM

The neural analysis models fNIRS network activity with a separate baseline and
task response for each state:

```text
z_n ~ Markov(K states)
y_n | z_n = k ~ N(A_k x_n + b_k, Sigma_k)
```

Here `y_n` is network activity, `x_n` contains the task regressors, `b_k` is the
state's baseline, and `A_k` gives its task-response coefficients.
The observations are neural signals, but the inputs include commission errors,
correct rejections, and reaction times. Later comparisons with behaviour are
therefore associations within the recorded data; they are not independent
held-out behavioural validation.

```python
from fnirs_glmhmm import neural

runs = neural.prepare_subject("sub-629")
y, x, meta = neural.stack(runs)  # (n_runs, T, 14) and (n_runs, T, 3)
```

The 14 features are seven network averages for each of two chromophores, HbO
and HbR. Limbic is excluded because of sparse coverage; TempPar is kept separate
from Default. Parcels must have HbO standard deviation above 1e-7 in every run.
The pipeline averages covered parcels by network, filters at 0.01-0.2 Hz,
resamples to 1 Hz, removes global-signal and DCT drift terms, and standardises
each run.

The three task inputs are `commission`, `correct_rejection`, and `rt_mod`,
convolved with a haemodynamic response function. A separate no-go regressor
would be redundant with the commission and correct-rejection regressors.

Each subject gets their own model. The script chooses K among the multi-state
candidates using leave-one-run-out likelihood and also scores a one-state
regression baseline. It still fits the selected multi-state model when the
baseline scores better; `cv_gain` records that comparison.

States are ordered by their `Default_HbO` baseline, allowing comparisons of
the highest- and lowest-baseline states across subjects. This ordering does not
establish that their full network profiles are equivalent.

Diagonal covariance is the default (`--cov diag`), estimating 14 variances per
state. A full covariance estimates 105 entries per state and can be explored
with `--cov full`.

```bash
uv run python scripts/fit_neural_hmm.py
uv run python scripts/neural_nulls.py
```

Both scripts write to `results/neural/`. Run the fit first: the nulls script
reads its saved models. The null comparisons use autoregressive and
phase-randomised surrogates to assess fitted structure, and circular shifts to
assess alignment between state probabilities and behaviour.
