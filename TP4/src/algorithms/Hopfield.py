import numpy as np

from utils.Hopfield.HopfieldParameters import HopfieldParameters
from utils.Hopfield.HopfieldResults import HopfieldResults


class Hopfield:

    def __init__(self, parameters: HopfieldParameters, patterns):
        self.patterns = np.transpose(patterns)
        self.w = (1 / len(patterns)) * np.matmul(self.patterns, np.transpose(self.patterns))
        np.fill_diagonal(self.w, 0)
        self.parameters = parameters

    def predict(self, x):
        stable = False
        i = 0
        results = []
        errors = []
        prev_state = np.sign(np.matmul(self.w, x))
        results.append(prev_state)

        while not stable and i < self.parameters.max_iterations:
            next_state = np.sign(np.matmul(self.w, prev_state))
            errors.append(next_state - prev_state)

            if np.array_equal(next_state, prev_state):
                stable = True
            else:
                results.append(next_state)
                prev_state = next_state

        return HopfieldResults(np.array(results), np.array(errors))
