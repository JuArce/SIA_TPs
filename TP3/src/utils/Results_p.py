import copy
from datetime import datetime

import numpy as np


class Results:

    def __init__(self, x: np.array, y: np.array, w: np.array, algorithm: str, function: str,
                 time: datetime, errors, max_error, iterations: int, std_devs=None):
        self.time = datetime.now() - time
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.w = copy.deepcopy(w)
        self.algorithm = algorithm
        self.function = function
        self.errors = errors
        self.iterations = iterations
        self.max_error = max_error
        self.std_devs = std_devs
