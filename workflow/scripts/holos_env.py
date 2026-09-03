import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import gymnasium as gym
from holos_pk import HolosPK


class HolosMulti(gym.env):
    def __init__(self, profile, episode_length, run_path):
        # power profile the reactor should follow
        self.profile = profile
        # length of a trajectory or episode
