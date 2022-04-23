from utils.Config_p import Config


class PerceptronParameters:
    def __init__(self, config: Config, activation_function, error_function, delta_function,
                 activation_function_derivative):
        self.perceptron = config.perceptron_algorithm
        self.cota = config.cota
        self.eta = config.eta
        self.tol_error = config.tol_error
        self.error_function = error_function
        self.betha = config.betha
        self.function = config.function
        self.activation_function = activation_function
        self.delta_function = delta_function
        self.activation_function_derivative = activation_function_derivative
