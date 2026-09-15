import numpy as np
import pandas as pd

from fnirs_glmhmm.behavior import signed_vtc, smooth_vtc, zone_labels


def test_smoothing_preserves_length_and_reduces_variance():
    rng = np.random.default_rng(0)
    vtc = rng.normal(size=500)
    smoothed = smooth_vtc(vtc)
    assert smoothed.shape == vtc.shape
    assert smoothed.var() < vtc.var()


def test_median_split_is_balanced():
    labels, thresh = zone_labels(np.arange(100, dtype=float))
    assert labels.sum() == 50
    assert thresh == 49.5


def test_explicit_threshold_is_respected():
    labels, thresh = zone_labels(np.arange(10, dtype=float), threshold=2.0)
    assert thresh == 2.0
    assert labels.sum() == 7


def events_frame(rt):
    return pd.DataFrame({"reaction_time": rt, "response_code": np.where(np.asarray(rt) > 0, 1, -1)})


def test_signed_vtc_matches_the_vtc_column_up_to_sign():
    rng = np.random.default_rng(0)
    rt = rng.lognormal(-0.4, 0.3, size=200)
    rt[[10, 11, 57]] = 0.0  # non-responses, as the event tables store them
    ev = events_frame(rt)

    z = (rt - rt.mean()) / rt.std(ddof=0)
    expected = z.copy()
    expected[[10, 11]] = z[9]
    expected[57] = z[56]

    assert np.allclose(signed_vtc(ev), expected)


def test_signed_vtc_splits_the_two_tails():
    rt = np.array([0.4, 0.4, 1.0, 1.0])
    signed = signed_vtc(events_frame(rt))
    assert np.all(signed[:2] < 0) and np.all(signed[2:] > 0)
    assert np.allclose(np.abs(signed[:2]), np.abs(signed[2:]))


def test_signed_vtc_back_fills_a_run_starting_on_a_non_response():
    rt = np.array([0.0, 0.5, 0.9])
    assert signed_vtc(events_frame(rt))[0] == signed_vtc(events_frame(rt))[1]
