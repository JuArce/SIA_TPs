import json


class Config:
    PERCEPTRON_ALGORITHMS = ["simple_perceptron",
                             "lineal_perceptron",
                             "not_linear_perceptron",
                             "multi_layer_perceptron"]

    def __init__(self, string):
        config = json.loads(string)
        self.perceptron_algorithm = config.get('perceptron_algorithm')
        self.cota = int(config.get('cota'))
        self.eta = float(config.get('eta'))
        self.betha = float(config.get('betha'))
        self.tol_error = float(config.get('tol_error'))
        self.function = config.get('function')

    def __str__(self):
        label = [self.perceptron_algorithm]
        return ''.join(label)
