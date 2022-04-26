from typing import Optional

import math

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
    return perceptron_parameters.betha * (1 - (sigmoide_tanh(n, perceptron_parameters) ** 2))


def sigmoide_logistic(n: float, perceptron_parameters: PerceptronParameters):
    return 1 / (1 + math.exp(-2 * n * perceptron_parameters.betha))


def sigmoide_logistic_derivative(n: float, perceptron_parameters: PerceptronParameters):
    return 2 * perceptron_parameters.betha * sigmoide_logistic(n, perceptron_parameters) * (
            1 - sigmoide_logistic(n, perceptron_parameters))
