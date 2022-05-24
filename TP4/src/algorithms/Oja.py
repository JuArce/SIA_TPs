import numpy as np

from utils.Oja.OjaParameters import OjaParameters


class Oja:

    def __init__(self, parameters: OjaParameters, variables_len):
        self.epochs = parameters.epochs
        self.learning_rate = parameters.learning_rate
        self.w = np.random.rand(variables_len)
        self.w = self.w / np.linalg.norm(self.w)

    def train(self, data):
        for epoch in range(self.epochs):
            for j in range(len(data)):
                s = self.w @ data[j]
                self.w = self.w + self.learning_rate * s * (data[j] - s * self.w)
        return self.w
