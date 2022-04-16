import copy
from datetime import datetime

import numpy as np

from utils.PerceptronParameters import PerceptronParameters


class Results:

    def __init__(self, x: np.array, y: np.array, w: np.array, parameters: PerceptronParameters, time: datetime,
                 iterations: int):
        self.time = datetime.now() - time
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.w = copy.deepcopy(w)
        self.parameters = parameters
        self.iterations = iterations
