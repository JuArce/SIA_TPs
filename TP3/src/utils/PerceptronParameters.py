from utils.Config_p import Config


class PerceptronParameters:
    def __init__(self, config: Config, activation_function, error_function):
        self.perceptron = config.perceptron_algorithm
        self.cota = config.cota
        self.eta = config.eta
        self.activation_function = activation_function
        self.error_function = error_function
        self.betha = config.betha
        self.function = config.function
