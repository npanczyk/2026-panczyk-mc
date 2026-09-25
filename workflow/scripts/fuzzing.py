from scipy.stats import norm

class System:
    def __init__(self, sensor, agent, environment):
        self.sensor = sensor
        self.agent = agent
        self.env = environment
        pass

class Disturbance:
    def __init__(self, xo, xa, xs):
        """
        Args:
            xo (float): Observation (sensor) disturbance
            xa (float): Agent disturbance
            xs (float): State (environment) disturbance
        """
        self.xo = xo
        self.xa = xa
        self.xs = xs
        pass

class DisturbanceDistribution:
    def __init__(self, Do, Da, Ds):
        """
        Args:
            Do (function): Observation (sensor) disturbance distribution
            Da (function): Agent disturbance distribution
            Ds (function): State (environment) distribution
        """
        self.Do = Do
        self.Da = Da
        self.Ds = Ds
        pass

class Do:
    "Observation distribution for the HolosMulti env. Requires standard deviations for Gaussian distributions of disturbances for power (p), change in power (dp), and drum angle (drum_angles)."
    def __init__(self, sigma_p=0.01, sigma_dp=0.02, sigma_drum=0.005):
        # initialize the disturbance distributions for the observation variables
        # DO NOT disturb pnext (assume the controller reads the prescribed power correctly)
        self.power_dd = norm(0, sigma_p),
        self.dp_dd = norm(0, sigma_dp),
        self.drum_dd = norm(0, sigma_drum)

    def sample(self, state=None):
        return {
            "p": self.power_dd.rvs(),
            "dp": self.dp_dd.rvs(),
            "drum_angles": self.drum_dd.rvs(size=8)
        }
