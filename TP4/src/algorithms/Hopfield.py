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
        energy = []
        results.append(x)
        prev_state = np.sign(np.matmul(self.w, x))
        prev_state = np.where(prev_state == 0, -1, prev_state)
        results.append(prev_state)

        while not stable and i < self.parameters.max_iterations:
            energy.append(self.calculate_energy(prev_state))
            next_state = np.sign(np.matmul(self.w, prev_state))
            next_state = np.where(next_state == 0, -1, next_state)

            if np.array_equal(next_state, prev_state):
                stable = True
            else:
                results.append(next_state)
                prev_state = next_state
            i += 1

        return HopfieldResults(np.array(results), np.array(energy))

    def calculate_energy(self, state):
        return -0.5 * np.matmul(np.matmul(state.T, self.w), state)
