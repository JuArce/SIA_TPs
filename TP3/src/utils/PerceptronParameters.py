from utils.Config_p import Config


class PerceptronParameters:
    def __init__(self, config: Config):
        self.algorithm = config.perceptron_algorithm
        self.cota = config.cota
        self.eta = config.eta
        self.betha = config.betha
        self.function = config.function
