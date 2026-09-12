import pytest
from holos_env import HolosMulti
import numpy as np
from scipy.interpolate import interp1d


@pytest.fixture
def env():
    return HolosMulti(profile=lambda t: 1, episode_length=10)


def test_init_observation_space_keys(env):
    assert set(env.observation_space.spaces.keys()) == {
        "dp",
        "p",
        "pnext",
        "drum_angles",
    }
    assert env.observation_space["drum_angles"].shape == (8,)


def test_init_action_space(env):
    assert env.action_space.shape == (8,)
    assert env.action_space.low[0] == -1
    assert env.action_space.high[0] == 1


def test_init_runtime():
    env = HolosMulti(profile=lambda t: 1, episode_length=10, dt=2)
    assert env.runtime == 20


def test_init_starting_state(env):
    assert env._dp == 0
    assert env._p == 1
    assert env._pnext == 1  # profile(0)
    np.testing.assert_array_equal(env._drum_angles, np.full(8, 77.8))


def test_mask_drums_training(env):
    env.mask_drums()
    assert env.masks.shape == (8,)
    assert env.masks.sum() == 8


def test_mask_drums_nontraining():
    env = HolosMulti(
        profile=lambda t: 1, episode_length=10, training=False, max_failed_drums=3
    )
    env.mask_drums()
    assert env.masks.shape == (8,)
    n_failed = 8 - env.masks.sum()
    assert n_failed in (0, 1, 2, 3)


def test_reset_observation(env):
    observation = env.reset()
    # remove numpy array for separate comparison
    drum_angles = observation.pop("drum_angles")
    assert observation == {
        "dp": 0,
        "p": 1,
        "pnext": 1,
    }
    np.testing.assert_array_equal(drum_angles, np.array([77.8] * 8))


def test_reset_history(env):
    env.reset()
    assert len(env.history) == 1
    row = env.history[0]
    # time, p, profile(time), 8 drum angles, 12 state values
    assert len(row) == 1 + 1 + 1 + 8 + 12


def test_log_history(env):
    env.reset()
    # this should initialize the history
    env._log_history()
    # this should append another row
    assert len(env.history) == 2


def test_get_dp_zero_when_power_unchanged(env):
    env.reset()
    env._p = env.history[-1][1]  # same as last logged power
    assert env._get_dp() == 0


def test_get_dp_nonzero_when_power_changes(env):
    env.reset()
    env._p = env.history[-1][1] + 0.5  # fudge to mimic ramp
    assert env._get_dp() == pytest.approx(0.5 / env.dt)


def test_render(env):
    env.reset()
    df = env.render()
    assert df["actual_power"][0] == 1, "steady state initial power value should be 100"
    assert df["drum_1"][0] == 77.8, "steady state initial drum angle should be 77.8"
