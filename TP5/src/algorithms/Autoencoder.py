import numpy as np
from scipy.optimize import minimize

from algorithms.Network import Network


class Autoencoder:

    def __init__(self, config, input_len, hidden_layers, latent_code_len):
        self.layers = self._build_hidden_layers(input_len, hidden_layers, latent_code_len)
        self.network = Network(config, self.layers)

    def train(self, data_x, data_y):
        aux = minimize(self.network.calculate_error, np.concatanate(self.network.weights),
                       args=(self.network.weights, data_x, data_y), method='Powell', bounds=None,
                       tol=None, callback=None,
                       options={'func': None, 'xtol': 0.0001, 'ftol': 0.0001, 'maxiter': None, 'maxfev': None,
                                'disp': False, 'direc': None, 'return_all': False})
        return None

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
