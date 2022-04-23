import copy
import random
from datetime import datetime

import numpy as np

from utils.PerceptronParameters import PerceptronParameters
from utils.Results_p import Results


def perceptron(perceptron_parameters: PerceptronParameters, x: np.array, y: np.array):
    time = datetime.now()
    i = 0
    w = np.zeros(len(x[0]))
    w_min = np.zeros(len(x[0]))
    error = 1
    error_min = 2 * len(x)
    cota = perceptron_parameters.cota
    eta = perceptron_parameters.eta

    while error > 0 and i < cota:
        norm = np.linalg.norm(w)
        if norm != 0:
            w = w / norm

        if i % 50000 == 0:
            print(i)

        idx = random.randint(0, len(x) - 1)

        h = x[idx] @ w  # producto interno (válida desde python 3.5)

        o = perceptron_parameters.activation_function(h, perceptron_parameters)
        delta_w = (eta * (y[idx] - o)) * x[idx]
        w = w + delta_w
        error = perceptron_parameters.error_function(x, y, w, len(x),perceptron_parameters.tol_error)

        if error < error_min:
            error_min = error
            w_min = copy.deepcopy(w)
        i += 1

    return Results(x, y, w_min, perceptron_parameters, time, i)
