import copy
import math
import random

import numpy as np

from utils.Kohonen.KohonenParameters import KohonenParameters
from utils.Kohonen.KohonenResults import KohonenResults


class Kohonen:

    def __init__(self, parameters: KohonenParameters, data):
        self.radius = parameters.initial_radius
        self.initial_radius = parameters.initial_radius
        self.epochs = parameters.epochs
        self.output_layer_len = parameters.output_layer_len
        self.initial_learning_rate = parameters.learning_rate
        self.learning_rate = parameters.learning_rate
        self.input_layer_len = len(data[0])
        # Se inicializan con los valores de las variables de los datos de entrada de forma random
        self.weights = self.initialize_weights(data)
        self.activation_map = np.empty((self.output_layer_len, self.output_layer_len), dtype=object)

    def train(self, data):

        # Itero todas las entradas 'epochs' veces
        for i in range(self.epochs):
            self.learning_rate = self.initial_learning_rate * math.exp(-i / self.epochs)
            self.radius = self.initial_radius * math.exp(-i / self.epochs)
            elem = data[random.randint(0, len(data) - 1)]
            idx = self.get_winner(elem)
            self.update_weights(idx, elem)

            # Se inicializan los pesos y se les asigna aleatoriamente los pesos de alguna de las entradas

    # Devuelve un array de k*k*n siendo n la dimensión de las entradas y k la cantidad de neuronas de la capa de salida

    def initialize_weights(self, data):
        weights = np.zeros((self.output_layer_len, self.output_layer_len, self.input_layer_len))
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                idx = random.randint(0, len(data) - 1)
                weights[i][j] = copy.deepcopy(data[idx])
        return weights

    def get_winner(self, data_i):
        distance = self.get_distance(data_i, self.weights)
        return np.asarray(np.unravel_index(np.argmin(distance, axis=None), distance.shape))

    def get_distance(self, data_i, weights):
        return np.linalg.norm(np.subtract(data_i, weights), axis=-1)

    def update_weights(self, idx, data_i):
        neighbors = self.get_neighbors(idx, len(self.weights), len(self.weights[0]), self.radius)

        for i in range(len(neighbors)):
            r = neighbors[i][0]
            c = neighbors[i][1]
            self.weights[r][c] = self.weights[r][c] + self.learning_rate * (data_i - self.weights[r][c])

    def get_neighbors(self, idx, rows, cols, radius):
        neighbors = []

        u_r = math.floor(idx[0] - radius) if math.floor(idx[0] - radius) > 0 else 0
        d_r = math.floor(idx[0] + radius) + 1 if math.floor(idx[0] + radius) + 1 < rows else rows

        l_c = math.floor(idx[1] - radius) if math.floor(idx[1] - radius) > 0 else 0
        r_c = math.floor(idx[1] + radius) + 1 if math.floor(idx[1] + radius) + 1 < cols else cols

        for i in range(u_r, d_r):
            for j in range(l_c, r_c):
                if math.dist(idx, [i, j]) <= radius:
                    neighbors.append([i, j])
        return neighbors

    def get_results(self, data, countries_name):
        # Agrupar países
        self.fill_activation_map(data, countries_name)

        # Analizar cantidad de elementos por neurona
        elements_per_neuron = self.get_elements_qty_per_neuron()

        # Distancia promedio entre neuronas vecinas
        weight_mean = self.get_weight_mean_neighbors()

        return KohonenResults(self.activation_map, elements_per_neuron, weight_mean, self.output_layer_len)

    # Retorna el promedio de la distancia entre neuronas vecinas
    def get_weight_mean_neighbors(self):
        w_mean = np.zeros((self.output_layer_len, self.output_layer_len))

        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                # el radio 1.5 permite obtener a las 8 neuronas vecinas
                neighbors = self.get_neighbors([i, j], len(self.weights), len(self.weights[0]), 1.5)
                aux = []
                for n in neighbors:
                    if i != n[0] or j != n[1]:
                        aux.append(np.linalg.norm(np.subtract(self.weights[i][j], self.weights[n[0]][n[1]])))
                w_mean[i][j] = np.average(aux)

        return w_mean

    def fill_activation_map(self, data, countries_name):
        for i in range(len(data)):
            idx = self.get_winner(data[i])
            if self.activation_map[idx[0]][idx[1]] is None:
                self.activation_map[idx[0]][idx[1]] = []
            self.activation_map[idx[0]][idx[1]].append(countries_name[i])

    def get_elements_qty_per_neuron(self):
        qty = np.zeros((self.output_layer_len, self.output_layer_len))
        for i in range(len(self.activation_map)):
            for j in range(len(self.activation_map[0])):
                qty[i][j] = len(self.activation_map[i][j]) if self.activation_map[i][j] is not None else 0

        return qty
