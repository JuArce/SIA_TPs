import math
import numpy as np


def sign(n: float):
    if n >= 0:
        return 1
    return -1


def get_error(x: np.array, y: np.array, w: np.array, p: int):
    i = 0
    ret = 0
    while i < p:
        o = x[i] @ w
        if not math.isclose(sign(o), y[i], abs_tol=0.00001):
            ret = ret + 1
        i = i + 1

    return ret
