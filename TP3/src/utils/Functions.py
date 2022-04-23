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


def sigmoide_logistic(n: float, perceptron_parameters: PerceptronParameters):
    return 1 / (1 + math.exp(-2 * n * perceptron_parameters.betha))


def get_error_sign(x: np.array, y: np.array, w: np.array, p: int, tol_error: float):
    i = 0
    ret = 0
    while i < p:
        o = x[i] @ w
        if not math.isclose(sign(o, None), y[i], abs_tol=tol_error):
            ret = ret + 1
        i = i + 1

    return ret


def get_error(x: np.array, y: np.array, w: np.array, p: int, tol_error: float):
    i = 0
    ret = 0
    while i < p:
        o = x[i] @ w
        if not math.isclose(o, y[i], abs_tol=tol_error):
            ret = ret + 1
        i = i + 1

    return ret
