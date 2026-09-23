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
        self.xs = xsb 
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