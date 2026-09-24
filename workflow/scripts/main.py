import pytest
from pke import HolosPK
import profiles
from scipy.interpolate import interp1d
import numpy as np
import viz


def run_demo():
    """Runs a demo case of a training and testing loop with minimal time_steps. Expected to perform terribly!"""
    training_kwargs, testing_kwargs = profiles.get_profile("train")
    run_folder = profiles.multi_drum_training(
        training_kwargs=training_kwargs, total_timesteps=50, n_envs=1, run_name="demo"
    )
    history = profiles.multi_drum_testing(
        testing_kwargs=testing_kwargs, run_folder=run_folder
    )
    viz.plot_power(history, save_dir=run_folder)
    return

def run_training():
    """Runs a full case of a training and testing loop."""
    training_kwargs, testing_kwargs = profiles.get_profile("train")
    run_folder = profiles.multi_drum_training(
        training_kwargs=training_kwargs, total_timesteps=int(5e6), n_envs=10, run_name="train_fivemillion"
    )
    history = profiles.multi_drum_testing(
        testing_kwargs=testing_kwargs, run_folder=run_folder
    )
    viz.plot_power(history, save_dir=run_folder)
    return


if __name__ == "__main__":
    run_training()
