"""GLM-HMM fitting on top of dynamax."""

import warnings
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from dynamax.hidden_markov_model import (
    BernoulliHMM,
    CategoricalRegressionHMM,
    LinearRegressionHMM,
    LogisticRegressionHMM,
)
from dynamax.hidden_markov_model.models.linreg_hmm import LinearRegressionHMMEmissions
from jax import vmap

EmissionKind = Literal["logistic", "categorical", "gaussian", "gaussian_diag", "bernoulli"]

VARIANCE_FLOOR = 1e-4


class _DiagonalEmissions(LinearRegressionHMMEmissions):
    """Linear regression emissions with the covariance constrained to diagonal.

    The M-step retains the fitted variances, floors them at VARIANCE_FLOOR,
    and zeros the off-diagonal entries. This uses D covariance parameters per
    state instead of D(D+1)/2.
    """

    def m_step(self, params, props, batch_stats, m_step_state):
        params, m_step_state = super().m_step(params, props, batch_stats, m_step_state)
        var = jnp.maximum(jnp.einsum("kii->ki", params.covs), VARIANCE_FLOOR)
        return params._replace(covs=vmap(jnp.diag)(var)), m_step_state


class DiagonalLinearRegressionHMM(LinearRegressionHMM):
    """LinearRegressionHMM with diagonal emission covariances."""

    def __init__(self, num_states, input_dim, emission_dim, **kwargs):
        super().__init__(num_states, input_dim, emission_dim, **kwargs)
        self.emission_component = _DiagonalEmissions(num_states, input_dim, emission_dim)


@dataclass
class Fit:
    """A fitted model plus the bookkeeping needed to compare fits."""

    model: object
    params: object
    props: object
    log_probs: np.ndarray
    num_states: int
    seed: int

    @property
    def final_log_prob(self) -> float:
        return float(self.log_probs[-1])

    def most_likely_states(self, emissions, inputs=None) -> np.ndarray:
        return np.asarray(
            self.model.most_likely_states(self.params, jnp.asarray(emissions), inputs=inputs)
        )

    def posterior(self, emissions, inputs=None):
        return self.model.smoother(self.params, jnp.asarray(emissions), inputs=inputs)

    def transition_matrix(self) -> np.ndarray:
        return np.asarray(self.params.transitions.transition_matrix)

    def marginal_log_prob(self, emissions, inputs=None) -> np.ndarray:
        """Log likelihood per sequence. Held-out data goes through here."""
        emissions = jnp.asarray(emissions)
        inputs = None if inputs is None else jnp.asarray(inputs)
        if emissions.ndim == 2:
            return np.asarray(self.model.marginal_log_prob(self.params, emissions, inputs))
        return np.asarray(
            vmap(lambda e, u: self.model.marginal_log_prob(self.params, e, u))(emissions, inputs)
        )


def build_model(
    kind: EmissionKind,
    num_states: int,
    input_dim: int,
    emission_dim: int = 1,
    num_classes: int = 2,
    stickiness: float = 0.0,
):
    """Pick the dynamax model that matches the observation you're predicting.

    logistic      - binary behaviour (e.g. in vs out of the zone)
    categorical   - discrete behaviour with >2 outcomes
    gaussian      - continuous emissions, e.g. parcel timeseries given a design matrix
    gaussian_diag - the same with diagonal covariances, for short sessions
    bernoulli     - binary emissions with no covariates (plain HMM, ignores inputs)
    """
    if kind == "logistic":
        return LogisticRegressionHMM(num_states, input_dim, transition_matrix_stickiness=stickiness)
    if kind == "categorical":
        return CategoricalRegressionHMM(
            num_states, num_classes, input_dim, transition_matrix_stickiness=stickiness
        )
    if kind == "gaussian":
        return LinearRegressionHMM(
            num_states, input_dim, emission_dim, transition_matrix_stickiness=stickiness
        )
    if kind == "gaussian_diag":
        return DiagonalLinearRegressionHMM(
            num_states, input_dim, emission_dim, transition_matrix_stickiness=stickiness
        )
    if kind == "bernoulli":
        return BernoulliHMM(num_states, emission_dim, transition_matrix_stickiness=stickiness)
    raise ValueError(f"unknown emission kind {kind!r}")


def fit_glm_hmm(
    emissions,
    inputs=None,
    num_states: int = 3,
    kind: EmissionKind = "logistic",
    num_restarts: int = 5,
    num_iters: int = 200,
    seed: int = 0,
    stickiness: float = 0.0,
    init_method: str = "prior",
    verbose: bool = False,
) -> Fit:
    """EM with several random inits; keeps the run with the best log likelihood.

    `emissions` and `inputs` are either (T, D) for one session or (N, T, D) for a
    batch of equal-length sessions. `init_method="kmeans"` seeds the emission
    means from a clustering of the data, which matters for continuous emissions
    where a prior draw can start far from the observed values.
    """
    emissions = jnp.asarray(emissions)
    inputs = None if inputs is None else jnp.asarray(inputs)

    input_dim = 0 if inputs is None else inputs.shape[-1]
    emission_dim = emissions.shape[-1] if emissions.ndim > 1 else 1
    num_classes = int(emissions.max()) + 1 if kind == "categorical" else 2

    # Build once: a fresh model object retraces and recompiles, which costs more
    # than the EM itself.
    model = build_model(kind, num_states, input_dim, emission_dim, num_classes, stickiness)
    init_kwargs = {"emissions": emissions} if init_method == "kmeans" else {}

    best = None
    for i in range(num_restarts):
        key = jr.PRNGKey(seed + i)
        params, props = model.initialize(key, method=init_method, **init_kwargs)
        params, lps = model.fit_em(
            params, props, emissions, inputs=inputs, num_iters=num_iters, verbose=verbose
        )
        fit = Fit(model, params, props, np.asarray(lps), num_states, seed + i)
        if best is None or fit.final_log_prob > best.final_log_prob:
            best = fit
    return best


def no_state_log_prob(
    y_train, x_train, y_test, x_test, diagonal: bool = True
) -> float:
    """Held-out log likelihood of the one-state baseline: a plain linear regression.

    dynamax cannot represent a one-state HMM (its Dirichlet transition prior needs
    at least two categories). This regression provides a baseline for assessing
    whether additional states improve held-out likelihood.
    """
    flat = lambda a: np.asarray(a).reshape(-1, np.asarray(a).shape[-1])  # noqa: E731
    ytr, xtr, yte, xte = flat(y_train), flat(x_train), flat(y_test), flat(x_test)

    design = np.column_stack([np.ones(len(xtr)), xtr])
    beta, *_ = np.linalg.lstsq(design, ytr, rcond=None)
    resid = ytr - design @ beta
    cov = resid.T @ resid / len(resid)
    if diagonal:
        cov = np.diag(np.maximum(np.diag(cov), VARIANCE_FLOOR))

    err = yte - np.column_stack([np.ones(len(xte)), xte]) @ beta
    sign, logdet = np.linalg.slogdet(cov)
    quad = np.einsum("ij,jk,ik->i", err, np.linalg.inv(cov), err)
    return float(-0.5 * (quad + logdet + err.shape[1] * np.log(2 * np.pi)).sum())


def state_sweep(
    emissions,
    inputs=None,
    state_range=(2, 3, 4, 5),
    **kwargs,
) -> dict[int, Fit]:
    """Fit the same data at several state counts so you can look at the LL curve."""
    return {k: fit_glm_hmm(emissions, inputs, num_states=k, **kwargs) for k in state_range}


def stack_sessions(arrays: list[np.ndarray]) -> np.ndarray:
    """Batch a list of (T, D) sessions into (N, T, D).

    dynamax needs equal-length batches, so ragged runs get truncated to the
    shortest one. If that is throwing away real data, fit the runs separately
    instead.
    """
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) > 1:
        t_min = min(lengths)
        warnings.warn(
            f"session lengths {sorted(lengths)} differ; truncating to {t_min}",
            stacklevel=2,
        )
        arrays = [a[:t_min] for a in arrays]
    return np.stack(arrays)


def add_bias(design: np.ndarray) -> np.ndarray:
    """Append a constant column. dynamax regression HMMs carry their own bias
    term, so only use this if you deliberately want a second one."""
    return np.column_stack([design, np.ones(len(design))])
