import numpy as np

from utils.Oja.OjaParameters import OjaParameters
from utils.Oja.OjaResults import OjaResults


class Oja:

    def __init__(self, parameters: OjaParameters, variables_len):
        self.epochs = parameters.epochs
        self.learning_rate = parameters.learning_rate
        self.w = np.random.rand(variables_len) * 2 - 1

    def train(self, data):
        w = []
        for epoch in range(self.epochs):
            for j in range(len(data)):
                w.append(self.w)
                s = self.w @ data[j]
                self.w = self.w + self.learning_rate * s * (data[j] - s * self.w)
        return OjaResults(w)
