"""HMM fits to VTC as an alternative to smoothed median-split zone labels.

The median split uses a fixed smoothing filter and a within-run threshold.
The HMM estimates emission distributions and transition probabilities, providing
both a decoded state path and state probabilities.

Pass unsmoothed VTC for the main comparison. Smoothing changes the temporal
dependence seen by the model. The reproduction notebook compares both choices.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from dynamax.hidden_markov_model import DiagonalGaussianHMM, GammaHMM

from fnirs_glmhmm.behavior import smooth_vtc, zone_labels


@dataclass
class ZoneFit:
    """A fitted zone HMM with states relabelled low VTC -> high VTC.

    State 0 has the lowest emission mean. For unsigned VTC this represents the
    smallest deviations; for signed VTC it represents the fastest relative responses.
    Sorting gives consistent labels, but does not make fitted parameters identical
    in meaning across subjects.
    """

    model: object
    params: object
    log_probs: np.ndarray
    num_states: int
    transform: str
    lengths: list[int]
    order: np.ndarray
    emissions: np.ndarray = field(repr=False)
    sort_dim: int = 0  # which emission dim defines "low -> high" when there are several

    @property
    def final_log_prob(self) -> float:
        return float(self.log_probs[-1])

    @property
    def transition_matrix(self) -> np.ndarray:
        p = np.asarray(self.params.transitions.transition_matrix)
        return p[np.ix_(self.order, self.order)]

    @property
    def state_means(self) -> np.ndarray:
        """Emission mean per state, ascending. (num_states,) if 1-D, else (num_states, dim)."""
        if self.transform == "gamma":
            conc = np.asarray(self.params.emissions.concentration)
            rate = np.asarray(self.params.emissions.rate)
            return (conc / rate)[self.order]
        m = np.asarray(self.params.emissions.means)[self.order]
        return m[:, 0] if m.shape[1] == 1 else m

    @property
    def expected_dwell(self) -> np.ndarray:
        """Expected run length in trials, 1 / (1 - p_ii)."""
        diag = np.diag(self.transition_matrix)
        return 1.0 / np.clip(1.0 - diag, 1e-12, None)

    def states(self) -> list[np.ndarray]:
        """Viterbi path per run, in the sorted state labelling."""
        inverse = np.argsort(self.order)
        return [inverse[self._viterbi(e)] for e in self._split(self.emissions)]

    def posteriors(self) -> list[np.ndarray]:
        """Smoothed state probabilities per run, (T, num_states)."""
        out = []
        for e in self._split(self.emissions):
            post = self.model.smoother(self.params, jnp.asarray(e))
            out.append(np.asarray(post.smoothed_probs)[:, self.order])
        return out

    def out_of_zone_prob(self) -> list[np.ndarray]:
        """P(highest-VTC state) per trial, the graded analogue of the binary label."""
        return [p[:, -1] for p in self.posteriors()]

    def occupancy(self) -> np.ndarray:
        """Fraction of trials spent in each state."""
        allstates = np.concatenate(self.states())
        return np.bincount(allstates, minlength=self.num_states) / len(allstates)

    def bic(self) -> float:
        n = sum(self.lengths)
        k = self.num_states - 1 + self.num_states * (self.num_states - 1)
        k += self.num_states if self.transform == "gamma" else 2 * self.num_states
        return float(k * np.log(n) - 2 * self.final_log_prob)

    def _viterbi(self, e) -> np.ndarray:
        return np.asarray(self.model.most_likely_states(self.params, jnp.asarray(e)))

    def _split(self, arr) -> list[np.ndarray]:
        bounds = np.cumsum(self.lengths)[:-1]
        return np.split(np.asarray(arr), bounds)


def prepare_vtc(vtc, transform: str = "log") -> np.ndarray:
    """Prepare one run's observations for the chosen emission model.

    log     - log1p then z-score
    zscore  - z-score without changing skewness
    gamma   - VTC values, with a small offset if any value is non-positive
    none    - pass through, for an already standardised series such as behavior.signed_vtc,
              where re-centring would move the fast/slow boundary off zero
    """
    v = np.asarray(vtc, dtype=float)
    if transform == "none":
        return v
    if transform == "log":
        v = np.log1p(v)
        return (v - v.mean()) / v.std()
    if transform == "zscore":
        return (v - v.mean()) / v.std()
    if transform == "gamma":
        if np.any(v <= 0):
            v = v + 1e-6
        return v
    raise ValueError(f"unknown transform {transform!r}")


def fit_zone_hmm(
    vtc_runs,
    num_states: int = 2,
    transform: str = "log",
    stickiness: float = 0.0,
    num_restarts: int = 10,
    num_iters: int = 200,
    seed: int = 0,
    verbose: bool = False,
) -> ZoneFit:
    """Fit a zone HMM to one run or to all of a subject's runs jointly.

    Runs share parameters and are concatenated to accommodate different lengths.
    This introduces one artificial transition per run boundary during fitting.
    Decoding is performed separately for each run.
    """
    if isinstance(vtc_runs, np.ndarray) and vtc_runs.ndim == 1:
        vtc_runs = [vtc_runs]
    elif not isinstance(vtc_runs, list | tuple):
        vtc_runs = [np.asarray(vtc_runs)]

    lengths = [len(np.asarray(v)) for v in vtc_runs]
    # Apply the selected transform separately to each run before concatenating.
    prepared = np.concatenate([prepare_vtc(v, transform) for v in vtc_runs])

    # GammaHMM takes scalar emissions; the Gaussian one wants a trailing dim.
    gamma = transform == "gamma"
    emissions = jnp.asarray(prepared if gamma else prepared[:, None])
    init_method = "prior" if gamma else "kmeans"

    # Reuse the model across restarts to avoid repeated JAX tracing and compilation.
    model = (
        GammaHMM(num_states, transition_matrix_stickiness=stickiness)
        if gamma
        else DiagonalGaussianHMM(num_states, 1, transition_matrix_stickiness=stickiness)
    )

    best = None
    for i in range(num_restarts):
        params, props = model.initialize(
            jr.PRNGKey(seed + i), method=init_method, emissions=emissions
        )
        params, lps = model.fit_em(params, props, emissions, num_iters=num_iters, verbose=verbose)
        lps = np.asarray(lps)
        if best is None or lps[-1] > best[1][-1]:
            best = (model, lps, params)

    model, lps, params = best
    fit = ZoneFit(
        model=model,
        params=params,
        log_probs=lps,
        num_states=num_states,
        transform=transform,
        lengths=lengths,
        order=np.arange(num_states),
        emissions=np.asarray(emissions),
    )
    fit.order = np.argsort(_raw_means(fit))
    return fit


def _raw_means(fit: ZoneFit) -> np.ndarray:
    if fit.transform == "gamma":
        conc = np.asarray(fit.params.emissions.concentration)
        return conc / np.asarray(fit.params.emissions.rate)
    return np.asarray(fit.params.emissions.means)[:, fit.sort_dim]


def fit_state_hmm(
    runs,
    num_states: int = 2,
    sort_dim: int = 0,
    stickiness: float = 0.0,
    num_restarts: int = 10,
    num_iters: int = 200,
    seed: int = 0,
) -> ZoneFit:
    """Gaussian HMM on multivariate per-trial features, one (n_trials, dim) array per run.

    Same machinery as fit_zone_hmm but for joint emissions, e.g. VTC together with
    pupil. Standardise the columns yourself; nothing is transformed here. States are
    sorted on `sort_dim`, so pass the column that means "more out of the zone".
    """
    runs = [np.atleast_2d(np.asarray(r, dtype=float)) for r in runs]
    runs = [r.T if r.shape[0] < r.shape[1] else r for r in runs]
    if any(np.isnan(r).any() for r in runs):
        raise ValueError("emissions contain NaN; dynamax has no missing-data support")

    dim = runs[0].shape[1]
    lengths = [len(r) for r in runs]
    emissions = jnp.asarray(np.concatenate(runs))

    model = DiagonalGaussianHMM(num_states, dim, transition_matrix_stickiness=stickiness)
    best = None
    for i in range(num_restarts):
        params, props = model.initialize(
            jr.PRNGKey(seed + i), method="kmeans", emissions=emissions
        )
        params, lps = model.fit_em(params, props, emissions, num_iters=num_iters, verbose=False)
        lps = np.asarray(lps)
        if best is None or lps[-1] > best[0][-1]:
            best = (lps, params)

    lps, params = best
    fit = ZoneFit(
        model=model,
        params=params,
        log_probs=lps,
        num_states=num_states,
        transform="zscore",
        lengths=lengths,
        order=np.arange(num_states),
        emissions=np.asarray(emissions),
        sort_dim=sort_dim,
    )
    fit.order = np.argsort(_raw_means(fit))
    return fit


def state_sweep(vtc_runs, state_range=(2, 3, 4), **kwargs) -> dict[int, ZoneFit]:
    """Fit the same observations at several state counts for comparison."""
    return {k: fit_zone_hmm(vtc_runs, num_states=k, **kwargs) for k in state_range}


def compare_to_median_split(fit: ZoneFit, vtc_runs, smooth_length: int = 20) -> dict:
    """Occupancy, switch counts and agreement against the smoothed median split.

    Agreement uses a binary label: the highest-VTC state versus all other states.
    Changing the state count can change how many trials enter the highest state.
    """
    if isinstance(vtc_runs, np.ndarray) and vtc_runs.ndim == 1:
        vtc_runs = [vtc_runs]

    hmm_states = fit.states()
    agree, hmm_frac, split_frac, hmm_sw, split_sw = [], [], [], [], []

    for states, vtc in zip(hmm_states, vtc_runs, strict=True):
        hmm_out = (states == fit.num_states - 1).astype(int)
        split_out, _ = zone_labels(smooth_vtc(vtc, length=smooth_length))
        agree.append(float((hmm_out == split_out).mean()))
        hmm_frac.append(float(hmm_out.mean()))
        split_frac.append(float(split_out.mean()))
        hmm_sw.append(int(np.diff(hmm_out).astype(bool).sum()))
        split_sw.append(int(np.diff(split_out).astype(bool).sum()))

    return {
        "agreement": float(np.mean(agree)),
        "agreement_per_run": agree,
        "frac_out_hmm": float(np.mean(hmm_frac)),
        "frac_out_median_split": float(np.mean(split_frac)),
        "switches_hmm": hmm_sw,
        "switches_median_split": split_sw,
        "expected_dwell": fit.expected_dwell.tolist(),
    }


def dwell_times(states: np.ndarray) -> dict[int, np.ndarray]:
    """Empirical run lengths per state, in trials."""
    states = np.asarray(states)
    change = np.flatnonzero(np.diff(states)) + 1
    segments = np.split(states, change)
    out: dict[int, list[int]] = {}
    for seg in segments:
        out.setdefault(int(seg[0]), []).append(len(seg))
    return {k: np.array(v) for k, v in sorted(out.items())}
