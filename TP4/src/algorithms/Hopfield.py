import numpy as np

from utils.HopfieldParameters import HopfieldParameters


class Hopfield:

    def __init__(self, parameters: HopfieldParameters, data):
        self.w = (1 / len(data)) * data.dot(np.transpose(data))
