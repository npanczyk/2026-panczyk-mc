import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import gymnasium as gym
from holos_pk import HolosPK
from pathlib import Path
import warnings


def scale(real_value, type):
    """Takes a value in real space and converts it to gym space

    Args:
        real_value (float): value in the real-scaled system
        type (string): Options: "power" or "dpower" or "dtheta" or "drum_angles"

    Returns:
        float: value scaled for the gym system
    """
    if type == "power":
        # real bounds are 0 to 22 MW, gym bounds are 0 to 1
        return real_value / 22
    elif type == "dpower":
        # real bounds are -22 to 22 MW/s # CHANGE THIS LATER USING PROMPT JUMP, gym bounds are -1 to 1
        return real_value / 22
    elif type == "dtheta":
        # real bounds are -0.5 deg/s to 0.5 deg/s, gym bounds are -1 to 1
        return real_value * 2
    elif type == "drum_angles":
        # real bounds are 0 to 180 deg, gym bounds are 0 to 1
        return real_value / 180


def descale(gym_value, type):
    """Takes a value in gym space and converts it to real space

    Args:
        gym_value (float): value in the gym-scaled system
        type (string): Options: "power" or "dpower" or "dtheta"

    Returns:
        float: value scaled for the gym system
    """
    if type == "power":
        # real bounds are 0 to 22 MW, gym bounds are 0 to 1
        return gym_value * 22
    elif type == "dpower":
        # real bounds are -22 to 22 MW/s # CHANGE THIS LATER USING PROMPT JUMP, gym bounds are -1 to 1
        return gym_value * 22
    elif type == "dtheta":
        # real bounds are -0.5 deg/s to 0.5 deg/s, gym bounds are -1 to 1
        return gym_value / 2
    elif type == "drum_angles":
        # real bounds are 0 to 180, gym bounds are 0 to 1
        return gym_value * 180


class HolosMulti(gym.Env):
    def __init__(
        self,
        profile,
        episode_length,
        training=True,
        max_failed_drums=1,
        dt=1,
        save_dir="./run_histories",
    ):
        """Initializes the HolosMulti environment

        Args:
            profile (interp1d function): callable function to yield the desired power (y) at any given time (x)
            episode_length (int): Number of timesteps in the episode (NOT seconds, this is just a count)
            training (bool): Whether the run is for training or testing.
            max_failed_drums (int): The maximum number of drums allowed to randomly fail.
            dt (int, optional): Number of seconds in a step. Defaults to 1 second.
            save_dir (str, optional): Directory to save the run.
        """
        self.profile = profile
        self.episode_length = episode_length
        self.training = training
        if training:
            self.max_failed_drums = 0
        else:
            self.max_failed_drums = max_failed_drums
        self.dt = dt
        self.save_dir = Path(save_dir)
        # make the save_dir a directory if it's not already
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            warnings.warn(
                f"'{self.save_dir}' is a file, not a directory. Falling back to default."
            )
            self.save_dir = Path("./run_histories")
            self.save_dir.mkdir(parents=True, exist_ok=True)
        # number of seconds in an episode
        self.runtime = episode_length * dt
        # initialize a point kinetics model
        self.pke = HolosPK()
        # initialize starting states
        self._dp = 0  # starting power rate of change, assume 0 at steady state
        self._p = (
            1  # starting power (MW), assume reactor starts at full power steady state
        )
        self._pnext = self.profile(0)  # next desired power (MW)
        self._drum_angles = np.array([77.8] * 8)

        self.observation_space = gym.spaces.Dict(
            {
                "dp": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "p": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "pnext": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "drum_angles": gym.spaces.Box(
                    low=0, high=1, shape=(8,), dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    def _get_observation(self):
        """Converts internal state to observation format (recommended per gym docs)

        Returns: dp, p, drum angles, pnext as a dictionary
        """
        return {
            "dp": self._dp,
            "p": self._p,
            "pnext": self._pnext,
            "drum_angles": scale(self._drum_angles, "drum_angles"),
        }

    def _get_dp(self):
        """Compute finite difference change in power. Must call this function BEFORE calling _log_history or the indexing will be off!

        Returns:
            float: change in power, dp/dt
        """
        self._plast = self.history[-1][1]  # last appended row, second column
        return (self._p - self._plast) / self.dt

    def _log_history(self, init=False):
        if init:
            self.history = [
                [
                    self.time,
                    self._p,  # observed power
                    self.profile(self.time),  # desired power
                    *self._drum_angles,
                    *self.state,
                ]
            ]
            return
        else:
            self.history.append(
                [
                    self.time,
                    self._p,  # observed power
                    self.profile(self.time),  # desired power
                    *self._drum_angles,
                    *self.state,
                ]
            )
            return

    def mask_drums(self):
        valid_maskings = tuple(range(self.max_failed_drums + 1))
        num_masks = np.random.choice(valid_maskings)
        self.masks = np.ones(8)
        mask_indices = np.random.choice(8, size=num_masks, replace=False)
        self.masks[mask_indices] = 0
        return

    def reset(self, seed=None):
        super().reset(seed=seed)
        # reset starting states
        self.time = 0
        self._dp = 0  # steady state start
        self._pnext = self.profile(
            0 + self.dt
        )  # desired power from profile at first time step
        self._drum_angles = np.array([77.8] * 8)
        self.state = self.pke.get_initial_conditions()
        # reminder for state shape: [n_r, c1, c2, c3, c4, c5, c6, Tf, Tm, Tc, xe, i]
        self._p, *_ = (
            self.state
        )  # gets initial power from pke neutron density (assumed equal in this code)
        self.mask_drums()
        observation = self._get_observation()
        self._log_history(init=True)
        return observation, {}  # return empty dict for info

    def step(self, action):
        # increment time by 1 step
        self.time += self.dt
        # take an action
        real_action = descale(action, "dtheta") * self.masks
        # get the linearly interpolated function for the drums
        drum_forcers = self.pke.drum_forcing(self._drum_angles, real_action)
        # update the environment state
        # solve PKE
        sol = solve_ivp(self.pke.reactor_dae, [0, 1], self.state, args=drum_forcers)
        self.state = sol.y[:, -1]
        # update the power level according to new neutron density from PKE
        self._p, *_ = self.state
        # get the change in power from the previous state
        self._dp = self._get_dp()
        # get the next desired power from the profile
        self._pnext = self.profile(self.time + self.dt)
        # update the drum angles
        self._drum_angles += real_action  # unscaled drum angles + real action
        self._drum_angles = np.clip(
            self._drum_angles, 0, 180
        )  # makes sure action leaves drums within physical bounds
        # get the observation
        observation = self._get_observation()
        # log the step in history
        self._log_history(init=False)

        # calculate the reward and termination criteria
        reward, terminated = self.calc_reward(
            current_power=self._p, desired_power=self.profile(self.time)
        )
        # check to see if time exceeded
        truncated = False
        if self.time >= self.runtime:
            truncated = True

        return (
            observation,
            reward,
            terminated,
            truncated,
            {},
        )  # return empty dict for info

    def calc_reward(self, current_power, desired_power):
        """Returns reward and whether the episode is terminated."""
        # First component: give reward to stay in the correct range
        diff = 100 * abs(current_power - desired_power)
        assert diff <= 200, "diff out of reasonable bounds"
        reward = 2 - diff

        # give a punish outside bounds if in train mode
        terminated = False
        if self.training and (
            diff > 5 or self._drum_angles.min() <= 0 or self._drum_angles.max() >= 180
        ):
            reward -= 100
            terminated = True

        return reward, terminated

    def render(self):
        """Converts the history list of lists into a dataframe."""
        run_history = np.array(self.history)
        column_names = [
            "time",
            "observed_power",
            "desired_power",
            "drum_1",
            "drum_2",
            "drum_3",
            "drum_4",
            "drum_5",
            "drum_6",
            "drum_7",
            "drum_8",
            "actual_power",
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
            "c6",
            "Tf",
            "Tm",
            "Tc",
            "Xe",
            "I",
        ]
        df = pd.DataFrame(run_history, columns=column_names)
        df["diff"] = (df["actual_power"] - df["desired_power"]) * 100

        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_path = self.save_dir / f"run_history_{timestr}.csv"
        df.to_csv(save_path, index=False)

        return df
