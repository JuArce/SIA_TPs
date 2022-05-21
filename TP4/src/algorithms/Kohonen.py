import copy
import math
import random

import numpy as np

from utils.KohonenParameters import KohonenParameters


class Kohonen:

    def __init__(self, parameters: KohonenParameters, data):
        self.radius = parameters.initial_radius
        self.epochs = parameters.epochs
        self.output_layer_len = parameters.output_layer_len
        self.learning_rate = parameters.learning_rate
        self.input_layer_len = len(data[0])
        self.weights = self.initialize_weights(data)

    def train(self, data):

        # Itero todas las entradas 'epochs' veces
        for epoch in range(self.epochs):
            self.learning_rate = self.learning_rate / (epoch+1)
            for j in range(len(data)):
                idx = self.get_winner(data[j])
                self.update_weights(idx, data[j])

    # Se inicializan los pesos y se les asigna aleatoriamente los pesos de alguna de las entradas
    # Devuelve un array de k*k*n siendo n la dimensión de las entradas y k la cantidad de neuronas de la capa de salida

    def initialize_weights(self, data):
        weights = np.zeros((self.output_layer_len, self.output_layer_len, self.input_layer_len))
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                idx = random.randint(0, len(data) - 1)
                weights[i][j] = copy.deepcopy(data[idx])
        return weights

    def get_winner(self, input):
        distance = self.get_distance(input, self.weights)
        return np.asarray(np.unravel_index(np.argmin(distance, axis=None), distance.shape))

    def get_distance(self, input, weights):
        return np.linalg.norm(np.subtract(input, weights), axis=-1)

    def update_weights(self, idx, input):
        neighbors = self.get_neighbors(idx, len(self.weights), len(self.weights[0]))

        for i in range(len(neighbors)):
            r = neighbors[i][0]
            c = neighbors[i][1]
            self.weights[r][c] = self.weights[r][c] + self.learning_rate * (input - self.weights[r][c])

    def get_neighbors(self, idx, rows, cols):
        neighbors = []

        u_r = math.floor(idx[0] - self.radius) if math.floor(idx[0] - self.radius) > 0 else 0
        d_r = math.floor(idx[0] + self.radius) if math.floor(idx[0] + self.radius) < rows else rows

        l_c = math.floor(idx[1] - self.radius) if math.floor(idx[1] - self.radius) > 0 else 0
        r_c = math.floor(idx[1] + self.radius) if math.floor(idx[1] + self.radius) < cols else cols

        for i in range(u_r, d_r):
            for j in range(l_c, r_c):
                if math.dist(idx, [i, j]) <= self.radius:
                    neighbors.append([i, j])
        return neighbors
