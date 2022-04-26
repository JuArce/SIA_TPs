import math
import numpy as np

from utils.PerceptronParameters import PerceptronParameters


def get_error(y: np.ndarray, o: np.ndarray, perceptron_parameters: PerceptronParameters):
    dim = len(y)
    ret = 0
    for i in range(dim):
        if not math.isclose(o[i], y[i], abs_tol=perceptron_parameters.tol_error):
            ret = ret + 1
    return ret
