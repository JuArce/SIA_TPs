from typing import Optional

import math
import numpy as np

from utils.PerceptronParameters import PerceptronParameters


def sign(n: float, perceptron_parameters: Optional['PerceptronParameters']):
    if n >= 0:
        return 1
    return -1


def identity(n: float, perceptron_parameters: Optional['PerceptronParameters']):
    return n


def sigmoide_tanh(n: float, perceptron_parameters: PerceptronParameters):
    return math.tanh(n * perceptron_parameters.betha)


def sigmoide_tanh_derivative(n: float, perceptron_parameters: PerceptronParameters):
    return perceptron_parameters.betha * (1 - (sigmoide_tanh(n, perceptron_parameters)) ** 2)


def sigmoide_logistic(n: float, perceptron_parameters: PerceptronParameters):
    return 1 / (1 + math.exp(-2 * n * perceptron_parameters.betha))


def sigmoide_logistic_derivative(n: float, perceptron_parameters: PerceptronParameters):
    return 2 * perceptron_parameters.betha * sigmoide_logistic(n, perceptron_parameters) * (
            1 - sigmoide_logistic(n, perceptron_parameters))


def delta_function(perceptron_parameters: PerceptronParameters, x, y, idx, h, o):
    return perceptron_parameters.eta * (y[idx] - o) * x[idx]


def delta_function_no_linear(perceptron_parameters: PerceptronParameters, x, y, idx, h, o):
    return perceptron_parameters.eta * (y[idx] - o) * \
           perceptron_parameters.activation_function_derivative(h, perceptron_parameters) * x[idx]


def get_error(x: np.array, y: np.array, w: np.array, p: int, perceptron_parameters: PerceptronParameters):
    i = 0
    ret = 0
    while i < p:
        o = perceptron_parameters.activation_function(x[i] @ w, perceptron_parameters)
        if not math.isclose(o, y[i], abs_tol=perceptron_parameters.tol_error):
            ret = ret + 1
        i = i + 1

    return ret
