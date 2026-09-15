import numpy as np
import pandas as pd
from test_neural import make_run

from fnirs_glmhmm import coupling


def posterior_from_states(states, k=2):
    """Hard posterior, so the coupling maths is checkable by hand."""
    post = np.zeros((len(states), k))
    post[np.arange(len(states)), states] = 1.0
    return post


def test_coupling_frame_reads_the_posterior_at_the_lagged_sample():
    run = make_run(n_samples=200)
    post = posterior_from_states(np.tile([0, 1], 100))
    df = coupling.coupling_frame([run], [post], lag_s=4.0)

    assert (df["sample"] == np.rint(df["trial"] * 0.8 + 4)).all()
    assert np.allclose(df["p1"].to_numpy(), post[df["sample"].to_numpy(), 1])
    assert set(df.columns) >= {"sub", "run", "trial", "commission", "omission", "log_rt", "vtc"}


def test_coupling_frame_drops_trials_past_the_window():
    run = make_run(n_samples=50, n_trials=100)
    df = coupling.coupling_frame([run], [posterior_from_states(np.zeros(50, int))], lag_s=4.0)
    assert len(df) < 100
    assert df["sample"].max() < 50


def test_log_rt_is_missing_on_no_press_trials():
    run = make_run(n_samples=200)
    df = coupling.coupling_frame([run], [posterior_from_states(np.zeros(200, int))])
    assert df.loc[df["omission"] == 1, "log_rt"].isna().all()
    assert coupling.subset(df, "log_rt")["log_rt"].notna().all()


def test_subset_picks_the_trials_the_outcome_is_defined_on():
    run = make_run(n_samples=200)
    df = coupling.coupling_frame([run], [posterior_from_states(np.zeros(200, int))])
    assert coupling.subset(df, "commission")["nogo"].all()
    assert not coupling.subset(df, "omission")["nogo"].any()


def test_subject_coupling_recovers_a_planted_slope():
    rng = np.random.default_rng(0)
    n = 400
    p = rng.random(n)
    df = pd.DataFrame(
        {
            "sub": "sub-000",
            "nogo": True,
            "go": False,
            "hit": False,
            "commission": 0.3 * p + rng.normal(0, 0.01, n),
            "p_dmn_high": p,
        }
    )
    rows = coupling.subject_coupling(df, ["p_dmn_high"], outcomes=["commission"])
    assert len(rows) == 1
    assert abs(rows["slope"].iloc[0] - 0.3) < 0.02
    assert rows["contrast"].iloc[0] > 0


def test_group_test_counts_signs_and_tests_them():
    rows = pd.DataFrame(
        {
            "sub": [f"sub-{i}" for i in range(10)],
            "outcome": "commission",
            "predictor": "p_dmn_high",
            "slope": [0.1, 0.2, 0.15, -0.05, 0.3, 0.12, 0.08, 0.22, 0.05, 0.18],
        }
    )
    out = coupling.group_test(rows)
    assert out["n_subjects"].iloc[0] == 10
    assert out["n_positive"].iloc[0] == 9
    assert out["p"].iloc[0] < 0.05


def test_group_test_needs_enough_subjects_to_report_a_p():
    rows = pd.DataFrame(
        {"sub": ["a", "b"], "outcome": "vtc", "predictor": "p_dmn_high", "slope": [0.1, 0.2]}
    )
    assert np.isnan(coupling.group_test(rows)["p"].iloc[0])


def test_slope_declines_to_guess_from_too_few_trials():
    assert np.isnan(coupling._slope(np.arange(5.0), np.arange(5.0)))
    assert np.isnan(coupling._slope(np.ones(50), np.arange(50.0)))
