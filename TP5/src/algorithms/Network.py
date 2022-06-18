import math
from copy import deepcopy
from statistics import mean

import numpy
import numpy as np
from numpy import random


class Network:
    LINEAR_FUNCTIONS = {
        "identity": {
            "f": lambda h: h,
            "fp": lambda h: 1
        },
    }

    NO_LINEAR_FUNCTIONS = {
        "logistic": {
            "f": lambda h, betha: 1 / (1 + math.exp(-2 * h * betha)),
            "fp": lambda h, betha: 2 * betha * (1 / (1 + math.exp(-2 * h * betha))) * (
                    1 - (1 / (1 + math.exp(-2 * h * betha))))
        },
        "tanh": {
            "f": lambda h, betha: math.tanh(h * betha),
            "fp": lambda h, betha: betha * (1 - (math.tanh(h * betha) ** 2))
        }
    }

    def __init__(self, config, layers):
        self.weights = self._build_weights(layers)
        if config.algorithm == 'no_linear_perceptron':
            self.act_function = Network.NO_LINEAR_FUNCTIONS[config.function]['f']
            self.act_function_derivative = Network.NO_LINEAR_FUNCTIONS[config.function]['fp']
        elif config.algorithm == 'linear_perceptron':
            self.act_function = Network.LINEAR_FUNCTIONS[config.function]['f']
            self.act_function_derivative = Network.LINEAR_FUNCTIONS[config.function]['fp']
        else:
            raise 'Invalid algorithm for multiperceptron'
        self.eta = config.eta
        self.cota = config.cota
        self.betha = config.betha
        self.algorithm = config.algorithm
        self.function = config.function
        self.max_error = config.max_error
        self.layers = layers
        self.w_by_layer = None

    def _build_weights(self, layers):
        weights = np.empty(len(layers) - 1, dtype=object)
        for i in range(len(weights)):
            weights[i] = np.random.uniform(-1, 1, size=(layers[i + 1], layers[i]))

        return weights

    def calculate_error(self, x, x_data, y_expected):

        errors = []
        weights = self.weights_resize(x)
        for i in range(len(x_data)):
            expected = y_expected[i]
            output = self.get_output(x_data[i], weights)
            errors.append((output - expected) ** 2)

        errors = np.array(errors)
        return mean((1 / len(x_data)) * sum(errors))

    def weights_resize(self, weights_array):

        weights = np.empty(len(self.layers) - 1, dtype=object)
        k = 0
        for i in range(len(weights)):
            weights[i] = np.empty((self.layers[i + 1], self.layers[i]))
            for j in range(len(weights[i])):
                size = len(weights[i][j])
                weights[i][j] = weights_array[k:k + len(weights[i][j])]
                k += size

        return weights

    def get_output(self, input_x, weights):
        x = deepcopy(input_x)
        for i in range(len(weights)):
            h = []
            for j in range(len(weights[i])):
                h.append(x @ weights[i][j])
            x = list(map(self.activation_function, h))
        return numpy.array(x)

    def activation_function(self, h):
        if self.algorithm == 'no_linear_perceptron':
            return self.act_function(h, self.betha)
        else:
            return self.act_function(h)

    def assign_weights(self, weights):
        self.weights = self.weights_resize(weights)
