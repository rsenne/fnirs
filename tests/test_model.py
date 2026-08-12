import numpy as np
import pytest

from fnirs_glmhmm.model import add_bias, fit_glm_hmm, stack_sessions


def test_recovers_two_states_from_synthetic_data():
    rng = np.random.default_rng(0)
    T = 800
    # Slow-switching latent state with opposite regression weights.
    z = np.repeat(rng.integers(0, 2, size=T // 40), 40)
    x = rng.normal(size=(T, 1))
    logit = np.where(z == 0, 3.0, -3.0) * x[:, 0]
    y = (rng.random(T) < 1 / (1 + np.exp(-logit))).astype(int)

    fit = fit_glm_hmm(y, x, num_states=2, kind="logistic", num_restarts=3, num_iters=60)
    est = fit.most_likely_states(y, inputs=x)

    agreement = max((est == z).mean(), (est != z).mean())
    assert agreement > 0.75


def test_stack_truncates_ragged_sessions():
    with pytest.warns(UserWarning):
        stacked = stack_sessions([np.zeros((10, 2)), np.zeros((7, 2))])
    assert stacked.shape == (2, 7, 2)


def test_add_bias_appends_ones():
    out = add_bias(np.zeros((5, 2)))
    assert out.shape == (5, 3)
    assert (out[:, -1] == 1).all()
