import numpy as np


class KohonenResults:

    def __init__(self, activation_map, elements_per_neuron, weight_mean, output_layer_len):
        self.activation_map = activation_map
        self.elements_per_neuron = elements_per_neuron
        self.weight_mean = weight_mean
        self.labels = np.empty((output_layer_len, output_layer_len), dtype=object)

        for i in range(len(self.activation_map)):
            for j in range(len(self.activation_map[i])):
                if self.labels[i][j] is None:
                    self.labels[i][j] = ""
                if self.activation_map[i][j] is not None:
                    for k in range(len(self.activation_map[i][j])):
                        self.labels[i][j] += self.activation_map[i][j][k]
                        self.labels[i][j] += "\n"
