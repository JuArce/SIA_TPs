from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from algorithms.Network import Network


class Autoencoder:
    step = 0

    def __init__(self, config, input_len, hidden_layers, latent_code_len):
        self.layers = self._build_hidden_layers(input_len, hidden_layers, latent_code_len)
        self.network = Network(config, self.layers)
        self.time = datetime.now()
        self.min_error = config.min_error
        self.learning_rate = config.learning_rate
        self.max_iter = config.max_iter
        self.iteration = 0

    def train(self, data_x, data_y):
        x = self.array_resize(self.network.weights)
        result = minimize(self.network.calculate_error, x, method='Powell',
                          args=(data_x, data_y),
                          jac=None, bounds=None,
                          tol=None,
                          callback=self.print_step,
                          options={'disp': True, 'maxiter': 10000})
        self.network.assign_weights(result.x)

    def encode(self, input_x):
        return self.network.get_output(input_x, self.network.weights[0:int(len(self.network.weights) / 2)])

    def decode(self, input_x_latent_code):
        return self.network.get_output(input_x_latent_code, self.network.weights[int(len(self.network.weights) / 2):])

    def _build_hidden_layers(self, input_len, hidden_layers, latent_code_len):
        layers = [input_len]
        layers.extend(hidden_layers)
        layers.append(latent_code_len)
        layers.extend(list(reversed(hidden_layers)))
        layers.append(input_len)
        return layers

    def array_resize(self, weights):
        x = np.array([])
        for w in weights:
            x = np.concatenate((x, w.flatten()), axis=0)
        return x

    def print_step(self, x):
        print('Iteration: ' + str(self.iteration))
        self.iteration += 1

    def get_output(self, input_x):
        return self.network.get_output(input_x, self.network.weights)
