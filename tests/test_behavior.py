import numpy as np

from fnirs_glmhmm.behavior import smooth_vtc, zone_labels


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
