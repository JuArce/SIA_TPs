import copy
from datetime import datetime

import math
import numpy as np
from numpy import random, vectorize, ndarray

from utils.PerceptronParameters import PerceptronParameters
from utils.Results_p import Results


class SimplePerceptron:

    def __init__(self, x: np.array, y: np.array, perceptron_parameters: PerceptronParameters):
        self.x = x
        self.y = y
        self.eta = perceptron_parameters.eta
        self.cota = perceptron_parameters.cota
        self.algorithm = perceptron_parameters.algorithm
        self.function = perceptron_parameters.function
        self.w = None

    def activation_function(self, h):
        if h >= 0:
            return 1
        return -1

    def delta_function(self, x: np.ndarray, y: np.ndarray, h: np.ndarray, o: np.ndarray):
        return self.eta * (y - o) * x

    def error_function(self, y: np.ndarray, o: np.ndarray):
        return (sum(y - o) ** 2) / 2

    def train_perceptron(self):
        time = datetime.now()
        i = 0
        w = np.zeros(len(self.x[0]))
        w_min = w
        error = 1
        error_min = 2 * len(self.x)

        while error > 0 and i < self.cota:

            idx = random.randint(0, len(self.x))
            h: ndarray = self.x @ w  # producto interno (válida desde python 3.5) Estado de excitacion
            o: ndarray = vectorize(pyfunc=self.activation_function)(h)  # Estado de Activacion
            delta_w = self.delta_function(self.x[idx], self.y[idx], h[idx], o[idx])
            w = w + delta_w
            error = self.error_function(self.y, o)

            if error < error_min:
                error_min = error
                w_min = copy.deepcopy(w)
            i += 1

        self.w = w_min

        return Results(self.x, self.y, self.w, self.algorithm, self.function, time, i)

    def test(self, x: np.ndarray, y: np.ndarray):

        h: ndarray = x @ self.w
        o: ndarray = vectorize(pyfunc=self.activation_function)(h)
        error = self.error_function(y, o)

        return error


class LinearPerceptron(SimplePerceptron):

    def __init__(self, x: np.ndarray, y: np.ndarray, perceptron_parameters: PerceptronParameters):
        super().__init__(x, y, perceptron_parameters)

    def activation_function(self, h):
        return h


class NoLinearPerceptron(SimplePerceptron):
    FUNCTIONS = {
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

    def __init__(self, x: np.ndarray, y: np.ndarray, perceptron_parameters: PerceptronParameters):
        y = 2 * (y - min(y)) / (max(y) - min(y)) - 1
        super().__init__(x, y, perceptron_parameters)
        self.betha = perceptron_parameters.betha
        self.act_function = NoLinearPerceptron.FUNCTIONS[perceptron_parameters.function]["f"]
        self.act_function_derivative = NoLinearPerceptron.FUNCTIONS[perceptron_parameters.function]["fp"]

    def activation_function(self, h):
        return self.act_function(h, self.betha)

    def delta_function(self, x: np.ndarray, y: np.ndarray, h: np.ndarray, o: np.ndarray):
        return self.eta * (y - o) * x * self.act_function_derivative(h, self.betha)
