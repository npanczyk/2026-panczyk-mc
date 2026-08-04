import numpy as np
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def clamp(number, min, max):
    if number < min:
        return min
    elif number > max:
        return max
    else:
        return number


class InvertedPendulum:
    def __init__(self, m, l, dt):
        self.m = m  # mass of the pendulum
        self.l = l  # length of the pendulum
        self.dt = dt  # time step in seconds
        self.g = 9.81  # m/s**2, gravitational constant
        self.w_max = 8.0  # maximum angular velocity
        self.a_max = 2.0  # maximum applicable torque
        self.Do = norm(0, 0.1)  # observation disturbance distribution
        self.alpha = np.array(
            [[-15, -8]]
        )  # (1 x 2) gain matrix for proportional control

    def get_initial_state(self):
        theta0 = np.random.uniform(-np.pi / 16, np.pi / 16)
        w0 = np.random.uniform(-1.0, 1.0)
        return theta0, w0

    def get_observation(self, s):
        # additive noise sensor
        # grab disturbance distribution from init and sample
        s = np.array([[s[0]], [s[1]]])
        noise = self.Do.rvs(size=s.shape)
        return s + noise

    def get_action(self, o):
        # proportional counter
        # take the dot product of the gain matrix and the observation
        torque = self.alpha @ o
        return torque[0][0]

    def get_state(self, s, a):
        """Gets values for the next state.

        Args:
            s (tuple): theta, w - current angle and angular velocity
            a (_type_): action, proposed torque from controller
        Returns:
            theta, omega: updated state variables
        """
        theta, w = s[0], s[1]
        dt, g, m, l = self.dt, self.g, self.m, self.l
        a = clamp(a, -1 * self.a_max, self.a_max)
        w = w + ((3 * g / (2 * l)) * np.sin(theta) + (3 * a) / (m * l**2)) * dt
        theta = theta + w * dt
        w = clamp(w, -1 * self.w_max, self.w_max)
        return theta, w

    def rollout(self, runtime):
        s = self.get_initial_state()  # initial state: (theta, omega)
        time = np.arange(0, runtime, self.dt)
        thetas = np.zeros(len(time))
        omegas = np.zeros(len(time))
        thetas[0] = s[0]
        omegas[0] = s[1]
        for i, t in enumerate(time):
            o = self.get_observation(s)
            a = self.get_action(o)
            s = self.get_state(s, a)
            thetas[i] = s[0]
            omegas[i] = s[1]
        return thetas, omegas

    def check_spec(self, trajectory):
        """Checks if a trajectory is within spec

        Args:
            trajectory (Numpy Array): array of theta values over course of rollout

        Returns:
            bool: True if all elements of the trajectory are within spec
        """
        return (np.abs(trajectory) <= np.pi / 4).all()


if __name__ == "__main__":
    n_rollouts = 1000
    pendulum = InvertedPendulum(m=1.0, l=1.0, dt=0.05)
    runtime = 10  # length of rollout in seconds
    timesteps = int(
        np.ceil(runtime / pendulum.dt)
    )  # this formula has edge cases, may need to change later to linspace in rollouts method
    successes = np.zeros(n_rollouts)
    theta_matrix = np.zeros((timesteps, n_rollouts))

    for r in range(n_rollouts):
        thetas, omegas = pendulum.rollout(runtime)
        theta_matrix[:, r] = thetas
        successes[r] = pendulum.check_spec(thetas)

    print(f"Success Rate: {successes.sum()/n_rollouts}")
    plt.plot(theta_matrix, color="steelblue", alpha=0.5, linewidth=0.8)
    plt.axhline(np.pi / 4, color="red", linewidth=1)
    plt.axhline(-np.pi / 4, color="red", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.savefig("../../results/inverted_pendulum.svg", dpi=300)
