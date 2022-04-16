import copy
import random
from datetime import datetime

import numpy as np

from utils.Functions import get_error
from utils.Functions import sign
from utils.PerceptronParameters import PerceptronParameters
from utils.Results_p import Results


def simple_perceptron(perceptron_parameters: PerceptronParameters, x: np.array, y: np.array):
    time = datetime.now()
    i = 0
    w = np.zeros(len(x[0]))
    w_min = np.zeros(len(x[0]))
    error = 1
    error_min = 2 * len(x)
    cota = perceptron_parameters.cota
    eta = perceptron_parameters.eta

    while error > 0 and i < cota:
        idx = random.randint(0, len(x) - 1)

        h = x[idx] @ w  # producto interno (válida desde python 3.5)
        o = sign(h)
        delta_w = (eta * (y[idx] - o)) * x[idx]
        w = w + delta_w
        error = get_error(x, y, w, len(x))

        if error < error_min:
            error_min = error
            w_min = copy.deepcopy(w)
        i += 1

    return Results(x, y, w_min, perceptron_parameters, time, i)
