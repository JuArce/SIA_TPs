import numpy as np
from numpy import random, matmul, mean, sum as npsum


class Network:
    LINEAR_FUNCTIONS = {
        "identity": {
            "f": lambda h: h,
            "fp": lambda h: 1
        },
    }

    NO_LINEAR_FUNCTIONS = {
        "logistic": {
            "f": lambda h, betha: 1 / (1 + np.exp(-2 * h * betha)),
            "fp": lambda h, betha: 2 * betha * (1 / (1 + np.exp(-2 * h * betha))) * (
                    1 - (1 / (1 + np.exp(-2 * h * betha))))
        },
        "tanh": {
            "f": lambda h, betha: np.tanh(h * betha),
            "fp": lambda h, betha: betha * (1 - (np.tanh(h * betha) ** 2))
        }
    }

    def __init__(self, config, layers):

        self.dimensions = []
        self.layers = layers
        self.weights = self._build_weights(self.layers)
        if config.algorithm == 'no_linear_perceptron':
            self.act_function = Network.NO_LINEAR_FUNCTIONS[config.function]['f']
            self.act_function_derivative = Network.NO_LINEAR_FUNCTIONS[config.function]['fp']
        elif config.algorithm == 'linear_perceptron':
            self.act_function = Network.LINEAR_FUNCTIONS[config.function]['f']
            self.act_function_derivative = Network.LINEAR_FUNCTIONS[config.function]['fp']
        else:
            raise 'Invalid algorithm for multiperceptron'
        self.betha = config.betha
        self.algorithm = config.algorithm
        self.function = config.function
        self.w_by_layer = None
        self.i = 0

    def _build_weights(self, layers):
        w = []
        for i in range(len(layers) - 1):
            self.dimensions.append((layers[i + 1], layers[i]))
            w.append(np.random.uniform(low=-1, high=1, size=(layers[i + 1], layers[i])))

        return np.array(w, dtype=object)

    def calculate_error(self, x, x_data, y_expected):

        output = []
        weights = self.weights_resize(x)
        for i in range(len(x_data)):
            output.append(self.get_output(x_data[i], weights))
        output = np.array(output)
        return mean((npsum((y_expected - output) ** 2, axis=1) / 2))

    def weights_resize(self, weights_array):

        weights = []
        idx = 0
        for dim in self.dimensions:
            flattened_dim = dim[0] * dim[1]
            weights.append(weights_array[idx:idx + flattened_dim].reshape(dim))
            idx += flattened_dim

        return weights

    def get_output(self, input_x, weights):
        activation_value = input_x.reshape((len(input_x), 1))
        for layer in range(len(weights)):
            activation_value = self.activation_function(matmul(weights[layer], activation_value))

        return activation_value.flatten()

    def activation_function(self, h):
        if self.algorithm == 'no_linear_perceptron':
            return self.act_function(h, self.betha)
        else:
            return self.act_function(h)

    def assign_weights(self, weights):
        self.weights = np.array(self.weights_resize(weights), dtype=object)
