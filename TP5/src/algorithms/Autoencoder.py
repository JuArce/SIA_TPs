import numpy as np
from scipy.optimize import minimize

from algorithms.Network import Network


class Autoencoder:

    def __init__(self, config, input_len, hidden_layers, latent_code_len):
        self.layers = self._build_hidden_layers(input_len, hidden_layers, latent_code_len)
        self.network = Network(config, self.layers)

    def array_resize(self, weights):
        x = np.array([])
        for w in weights:
            x = np.concatenate((x, w.flatten()), axis=0)
        return x

    def train(self, data_x, data_y):
        x = self.array_resize(self.network.weights)
        result = minimize(self.network.calculate_error, x, method='Powell',
                          args=(data_x, data_y),
                          jac=None, bounds=None,
                          tol=None,
                          callback=None,
                          options={'disp': True, 'maxiter': 15000})
        self.network.assign_weights(result.x)


    def encode(self):
        return None

    def decode(self):
        return None

    def _build_hidden_layers(self, input_len, hidden_layers, latent_code_len):
        layers = [input_len]
        layers.extend(hidden_layers)
        layers.append(latent_code_len)
        layers.extend(list(reversed(hidden_layers)))
        layers.append(input_len)
        return layers
