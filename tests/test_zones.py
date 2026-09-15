import numpy as np
import pytest

from fnirs_glmhmm.zones import (
    compare_to_median_split,
    dwell_times,
    fit_zone_hmm,
    prepare_vtc,
    state_sweep,
)


def synthetic_vtc(rng, n_blocks=25, block=24):
    """Slow two-state VTC: in-zone is low and stable, out-of-zone high and variable."""
    z = np.repeat(rng.integers(0, 2, size=n_blocks), block)
    vtc = np.where(
        z == 0,
        rng.lognormal(-0.5, 0.4, size=len(z)),
        rng.lognormal(0.6, 0.7, size=len(z)),
    )
    return vtc, z


def test_recovers_synthetic_zone_states():
    rng = np.random.default_rng(0)
    vtc, z = synthetic_vtc(rng)
    fit = fit_zone_hmm(vtc, num_states=2, num_restarts=5, num_iters=100)
    assert (fit.states()[0] == z).mean() > 0.9


def test_states_are_ordered_by_vtc():
    rng = np.random.default_rng(0)
    vtc, _ = synthetic_vtc(rng)
    for transform in ("log", "zscore", "gamma", "none"):
        fit = fit_zone_hmm(vtc, num_states=3, transform=transform, num_restarts=3, num_iters=60)
        means = fit.state_means
        assert np.all(np.diff(means) > 0), f"{transform}: {means}"


def test_ordering_is_consistent_across_states_and_posteriors():
    rng = np.random.default_rng(2)
    vtc, _ = synthetic_vtc(rng)
    fit = fit_zone_hmm(vtc, num_states=2, num_restarts=3, num_iters=80)
    states, post = fit.states()[0], fit.posteriors()[0]
    # Viterbi label and argmax of the reordered posterior should mostly agree.
    assert (states == post.argmax(1)).mean() > 0.95
    assert vtc[states == 1].mean() > vtc[states == 0].mean()


def test_multiple_ragged_runs_split_back_correctly():
    rng = np.random.default_rng(1)
    runs = [synthetic_vtc(rng, n_blocks=20)[0] for _ in range(3)]
    runs[2] = runs[2][:300]
    fit = fit_zone_hmm(runs, num_states=2, num_restarts=2, num_iters=60)
    assert fit.lengths == [len(r) for r in runs]
    assert [len(s) for s in fit.states()] == fit.lengths
    assert [p.shape for p in fit.posteriors()] == [(n, 2) for n in fit.lengths]


def test_out_of_zone_prob_is_a_probability():
    rng = np.random.default_rng(3)
    vtc, _ = synthetic_vtc(rng)
    fit = fit_zone_hmm(vtc, num_states=2, num_restarts=2, num_iters=60)
    p = fit.out_of_zone_prob()[0]
    assert p.shape == vtc.shape
    assert ((p >= 0) & (p <= 1)).all()


def test_hmm_switches_less_than_median_split():
    rng = np.random.default_rng(0)
    vtc, _ = synthetic_vtc(rng)
    fit = fit_zone_hmm(vtc, num_states=2, num_restarts=5, num_iters=100)
    cmp = compare_to_median_split(fit, [vtc])
    assert cmp["switches_hmm"][0] < cmp["switches_median_split"][0]
    assert 0.0 <= cmp["agreement"] <= 1.0


def test_state_sweep_returns_one_fit_per_k():
    rng = np.random.default_rng(4)
    vtc, _ = synthetic_vtc(rng, n_blocks=10)
    fits = state_sweep(vtc, state_range=(2, 3), num_restarts=2, num_iters=40)
    assert sorted(fits) == [2, 3]
    assert all(np.isfinite(f.bic()) for f in fits.values())


def test_dwell_times_counts_runs():
    d = dwell_times(np.array([0, 0, 0, 1, 1, 0]))
    assert list(d[0]) == [3, 1]
    assert list(d[1]) == [2]


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        prepare_vtc(np.ones(10), transform="nope")
