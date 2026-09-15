import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fnirs_glmhmm import neural
from fnirs_glmhmm.neural import RunData


def fake_events(n=100, onset0=10.0, nogo_every=10, seed=0):
    rng = np.random.default_rng(seed)
    onset = onset0 + 0.8 * np.arange(n)
    trial_type = np.where(np.arange(n) % nogo_every == 0, "mnt", "city")
    rc = np.where(trial_type == "mnt", 0, 1)
    rc[::20] = -2  # every other no-go is a commission error
    rt = np.where(rc == 1, rng.uniform(0.3, 1.0, n), 0.0)
    rt[rc == -2] = rng.uniform(0.3, 1.0, (rc == -2).sum())
    return pd.DataFrame(
        {
            "onset": onset,
            "trial_type": trial_type,
            "response_code": rc,
            "reaction_time": rt,
            "VTC": rng.gamma(2, 0.5, n),
        }
    )


def fake_parcel_run(parcels, n_time=900, dead=(), seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(2, len(parcels), n_time))
    for i in dead:
        data[:, i] = 1e-15 * rng.normal(size=(2, n_time))
    return xr.DataArray(
        data,
        dims=("chromo", "parcel", "time"),
        coords={"chromo": ["HbO", "HbR"], "parcel": list(parcels), "time": np.arange(n_time) / 9.0},
    )


def test_network_labels_collapse_17_to_7():
    parcels = ["DefaultB_PFCd_1_LH", "VisPeri_ExStr_2_RH", "ContA_IPS_1_LH", "TempPar_1_RH"]
    assert list(neural.network_labels(parcels)) == ["Default", "Vis", "Cont", "TempPar"]
    assert neural.network_labels(parcels, temppar="default")[-1] == "Default"


def test_covered_mask_drops_parcels_outside_the_probe():
    parcels = [f"DefaultA_p{i}_LH" for i in range(5)]
    runs = [fake_parcel_run(parcels, dead=(1, 3), seed=s) for s in range(2)]
    assert list(neural.covered_mask(runs)) == [True, False, True, False, True]


def test_covered_mask_needs_coverage_in_every_run():
    parcels = [f"DefaultA_p{i}_LH" for i in range(3)]
    runs = [fake_parcel_run(parcels, dead=(), seed=0), fake_parcel_run(parcels, dead=(2,), seed=1)]
    assert list(neural.covered_mask(runs)) == [True, True, False]


def test_hrf_peaks_where_a_canonical_hrf_should():
    fs = 9.0
    h = neural.hrf_kernel(fs)
    peak_s = float(np.argmax(h)) / fs
    assert 4.0 < peak_s < 7.0
    assert h.min() < 0  # the undershoot survives normalisation


def test_design_omits_nogo_and_stays_full_rank():
    events = fake_events()
    t = np.arange(0, 200, 1 / 9.0)
    x = neural.design_matrix(events, t, 9.0)

    assert x.shape == (len(t), len(neural.INPUT_NAMES))
    assert np.linalg.matrix_rank(x) == x.shape[1]
    # The regressor the plan asked for is exactly the sum of two of these, which is
    # why it is not a column: adding it would make the design singular.
    nogo = x[:, 0] + x[:, 1]
    assert np.linalg.matrix_rank(np.column_stack([x, nogo])) == x.shape[1]


def test_rt_modulation_uses_hits_only():
    events = fake_events()
    t = np.arange(0, 200, 1 / 9.0)
    x = neural.design_matrix(events, t, 9.0)
    # z-scored amplitudes sum to zero, so the modulated column has both signs.
    assert x[:, 2].min() < 0 < x[:, 2].max()


def test_bandpass_removes_drift_and_keeps_the_task_band():
    fs, n = 9.0, 3000
    t = np.arange(n) / fs
    drift = 3 * t / t[-1]
    signal = np.sin(2 * np.pi * 0.05 * t)
    out = neural.bandpass(np.atleast_2d(drift + signal), fs)[0]
    assert np.corrcoef(out[200:-200], signal[200:-200])[0, 1] > 0.98


def test_resample_window_lands_on_the_requested_grid():
    t = np.arange(0, 400, 1 / 9.0)
    x = np.atleast_2d(t)  # a ramp, so interpolation is exact
    out = neural.resample_window(x, t, t0=10.0, window_s=100.0, target_fs=1.0)
    assert out.shape == (1, 100)
    assert np.allclose(out[0], 10.0 + np.arange(100), atol=1e-6)


def test_dct_basis_stops_at_the_cutoff():
    basis = neural.dct_basis(355, dt=1.0, cutoff=0.01)
    assert basis.shape == (355, 7)
    assert np.allclose(basis.mean(axis=0), 0, atol=0.02)


def test_regress_out_removes_the_nuisance():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 2))
    y = z @ np.array([[1.0, -2.0], [0.5, 3.0]]) + 0.01 * rng.normal(size=(200, 2))
    resid = neural.regress_out(y, z)
    assert np.abs(np.corrcoef(resid[:, 0], z[:, 0])[0, 1]) < 1e-8
    assert resid.std() < 0.05


def make_run(n_samples=100, t0=10.0, n_trials=100):
    return RunData(
        sub="sub-000",
        run=1,
        y=np.zeros((n_samples, 2)),
        x=np.zeros((n_samples, 3)),
        t0=t0,
        target_fs=1.0,
        n_covered=10,
        feature_names=["Default_HbO", "Default_HbR"],
        input_names=list(neural.INPUT_NAMES),
        events=fake_events(n=n_trials, onset0=t0),
    )


def test_trial_samples_respects_the_lag_and_the_window():
    run = make_run(n_samples=50)
    idx = neural.trial_samples(run, lag_s=4.0)
    assert idx[0] == 4  # first trial sits at t0, so it lands at the lag
    assert idx[-1] == -1  # 100 trials at 0.8 s overrun a 50 s window
    assert (idx[idx >= 0] < 50).all()


def test_trial_samples_shift_with_the_lag():
    run = make_run(n_samples=200)
    early = neural.trial_samples(run, lag_s=2.0)
    late = neural.trial_samples(run, lag_s=6.0)
    assert (late[:10] - early[:10] == 4).all()


def test_stack_batches_runs():
    runs = [make_run(), make_run()]
    y, x, meta = neural.stack(runs)
    assert y.shape == (2, 100, 2)
    assert x.shape == (2, 100, 3)
    assert list(meta.columns) == ["sub", "run", "n_covered"]


def test_zscore_is_per_column():
    x = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    out = neural.zscore(x)
    assert np.allclose(out.mean(axis=0), 0)
    assert np.allclose(out[:, 0], out[:, 1])


def test_zscore_survives_a_constant_column():
    out = neural.zscore(np.column_stack([np.ones(10), np.arange(10.0)]))
    assert not np.isnan(out).any()


@pytest.mark.parametrize("temppar", ["separate", "default"])
def test_feature_count(temppar):
    n_networks = len(neural.FEATURE_NETWORKS) - (temppar == "default")
    assert n_networks * len(neural.CHROMOPHORES) in (12, 14)
