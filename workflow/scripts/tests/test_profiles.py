import pytest
from pke import HolosPK
import profiles
from scipy.interpolate import interp1d
import numpy as np


def test_get_profile():
    train_kwargs, test_kwargs = profiles.get_profile("train", 2)
    expected_profile = interp1d(
        [0, 10, 20, 30, 50, 70, 120, 140, 160, 195, 200],
        [1, 1, 0.98, 0.99, 0.80, 0.60, 0.60, 0.70, 0.70, 0.80, 0.80],
    )
    sample_times = [0, 15, 60, 130, 190, 200]

    # check that the training profile is as expected
    assert np.allclose(
        train_kwargs["profile"](sample_times), expected_profile(sample_times)
    )
    # check that the training profile matches the testing profile, per the profile type
    assert np.allclose(
        train_kwargs["profile"](sample_times), test_kwargs["profile"](sample_times)
    )
    # check some other parameters
    assert train_kwargs["dt"] == 1
    assert train_kwargs["training"] == True
    assert test_kwargs["training"] == False
    assert train_kwargs["max_failed_drums"] == 0
    assert test_kwargs["max_failed_drums"] == 2


########## quick integration testing ##########
@pytest.fixture
def quick_kwargs():
    training_kwargs, testing_kwargs = profiles.get_profile()
    for kwarg in [training_kwargs, testing_kwargs]:
        kwarg["episode_length"] = 5
    return training_kwargs, testing_kwargs


def test_multi_drum_runthrough(quick_kwargs, tmp_path, monkeypatch):
    training_kwargs, testing_kwargs = quick_kwargs
    monkeypatch.chdir(tmp_path)
    run_folder = profiles.multi_drum_training(
        training_kwargs=training_kwargs, total_timesteps=50, n_envs=1
    )
    assert (run_folder / "models" / "best_model.zip").exists()

    history = profiles.multi_drum_testing(testing_kwargs)
    assert len(history) > 0
    assert "actual_power" in history.columns
