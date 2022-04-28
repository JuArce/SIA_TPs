import copy
from datetime import datetime
from typing import Optional

import math
import numpy as np
from numpy import random, vectorize, ndarray

from utils.PerceptronParameters import PerceptronParameters
from utils.Results_p import Results


class SimplePerceptron:

    def __init__(self, x: np.array, y: np.array, perceptron_parameters: PerceptronParameters):
        self.x = x
        self.y = y
        self.eta = perceptron_parameters.eta
        self.cota = perceptron_parameters.cota
        self.algorithm = perceptron_parameters.algorithm
        self.function = perceptron_parameters.function
        self.w = None

    def activation_function(self, h):
        if h >= 0:
            return 1
        return -1

    def delta_function(self, x: np.ndarray, y: np.ndarray, h: np.ndarray, o: np.ndarray):
        return self.eta * (y - o) * x

    def error_function(self, y: np.ndarray, o: np.ndarray):
        return (sum(y - o) ** 2) / 2

    def train_perceptron(self):
        time = datetime.now()
        i = 0
        w = np.zeros(len(self.x[0]))
        w_min = w
        error = 1
        error_min = 2 * len(self.x)

        while error > 0 and i < self.cota:

            idx = random.randint(0, len(self.x))
            h: ndarray = self.x @ w  # producto interno (válida desde python 3.5) Estado de excitacion
            o: ndarray = vectorize(pyfunc=self.activation_function)(h)  # Estado de Activacion
            delta_w = self.delta_function(self.x[idx], self.y[idx], h[idx], o[idx])
            w = w + delta_w
            error = self.error_function(self.y, o)

            if error < error_min:
                error_min = error
                w_min = copy.deepcopy(w)
            i += 1

        self.w = w_min

        return Results(self.x, self.y, self.w, self.algorithm, self.function, time, i)

    def test(self, x: np.ndarray, y: np.ndarray):

        h: ndarray = x @ self.w
        o: ndarray = vectorize(pyfunc=self.activation_function)(h)
        error = self.error_function(y, o)

        return error


class LinearPerceptron(SimplePerceptron):

    def __init__(self, x: np.ndarray, y: np.ndarray, perceptron_parameters: PerceptronParameters):
        super().__init__(x, y, perceptron_parameters)

    def activation_function(self, h):
        return h


class NoLinearPerceptron(SimplePerceptron):
    FUNCTIONS = {
        "logistic": {
            "f": lambda h, betha: 1 / (1 + math.exp(-2 * h * betha)),
            "fp": lambda h, betha: 2 * betha * (1 / (1 + math.exp(-2 * h * betha))) * (
                    1 - (1 / (1 + math.exp(-2 * h * betha))))
        },
        "tanh": {
            "f": lambda h, betha: math.tanh(h * betha),
            "fp": lambda h, betha: betha * (1 - (math.tanh(h * betha) ** 2))
        }
    }

    def __init__(self, x: np.ndarray, y: np.ndarray, perceptron_parameters: PerceptronParameters):
        y = 2 * (y - min(y)) / (max(y) - min(y)) - 1
        super().__init__(x, y, perceptron_parameters)
        self.betha = perceptron_parameters.betha
        self.act_function = NoLinearPerceptron.FUNCTIONS[perceptron_parameters.function]["f"]
        self.act_function_derivative = NoLinearPerceptron.FUNCTIONS[perceptron_parameters.function]["fp"]

    def activation_function(self, h):
        return self.act_function(h, self.betha)

    def delta_function(self, x: np.ndarray, y: np.ndarray, h: np.ndarray, o: np.ndarray):
        return self.eta * (y - o) * x * self.act_function_derivative(h, self.betha)


class MultiPerceptron:

    def __init__(self, x: np.array, y: np.array, perceptron_parameters: PerceptronParameters):
        self.x = x
        self.y = y
        self.eta = perceptron_parameters.eta
        self.cota = perceptron_parameters.cota
        self.algorithm = perceptron_parameters.algorithm
        self.function = perceptron_parameters.function
        self.layers = perceptron_parameters.layers
        self.max_error = perceptron_parameters.max_error
        self.len_layers = len(self.layers)

        self.perceptrons: [[Perceptron]] = []

        for i in range(self.len_layers):

            if i == 0:
                self.perceptrons.append(
                    self.create_layer(self.layers[i] + 1, None, False))  # si es el primero no le llega ningun peso
            elif i == self.len_layers - 1:
                self.perceptrons.append(self.create_layer(self.layers[i], np.zeros(self.layers[
                                                                                       i - 1] + 1),
                                                          True))  # si es el último tiene la cantidad de pesos de la capa inferior + 1 por el umbral
            else:
                self.perceptrons.append(self.create_layer(self.layers[i] + 1, np.zeros(self.layers[i - 1] + 1), False))

    def train(self):
        time = datetime.now()
        error = 1
        while error > self.max_error:
            idx = random.randint(0, len(self.x))

            # Le asigno a la capa de entrada los valores de entrada
            for i in range(len(self.perceptrons[0])):
                self.perceptrons[0][i].o = self.x[idx][i]
                self.perceptrons[0][i].h = self.x[idx][i]

            # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
            for m in range(1, self.len_layers):  # por cada capa 1 a M
                for i in range(self.layers[m]):  # recorro todos los de la capa actual menos el del umbral
                    self.perceptrons[m][i].h = self.calculate_h(self.perceptrons[m][i].w, self.perceptrons[
                        m - 1])  # le paso los pesos que me llegan y la capa anterior
                    self.perceptrons[m][i].o = self.activation_function(self.perceptrons[m][i].h)

            # Calculo d en la capa de salida

            for i in range(len(self.perceptrons[self.len_layers - 1])):  # recorro los perceptrones de la capa de salida
                self.perceptrons[self.len_layers - 1][i].d = self.activation_function_derivative(
                    self.perceptrons[self.len_layers - 1][i].h) * \
                                                             (self.y[idx] - self.perceptrons[self.len_layers - 1][
                                                                 i].o)  # TODO: ver si recibe otro Yi que pasa

            # Retropropagar hacia abajo

            for m in range(self.len_layers - 1, 1, -1):  # retropropagar de la capa de salida a la anteultima
                for i in range(len(self.perceptrons[m - 1])):
                    h = self.perceptrons[m - 1][i].h
                    self.perceptrons[m - 1][i].d = self.activation_function_derivative(h) * self.calculate_d(
                        self.perceptrons[m], idx)

            # Calculando los nuevos pesos

            for m in range(self.len_layers - 1, 1, -1):
                for i in range(len(self.perceptrons[m])):
                    self.perceptrons[m][i].w = self.perceptrons[m][i].w + self.calculate_delta_w(
                        self.perceptrons[m][i].d, self.perceptrons[m - 1])

            errors = []
            #Calculo las funciones de activacion con toods los
            # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
            for j in range(len(self.x)):
                pass
                # for m in range

        return Results(self.x, self.y, self.w, self.algorithm, self.function, time)

    def predict(self):
        return None

    def activation_function(self, h):
        return h

    def activation_function_derivative(self, h):
        return 1

    def calculate_h(self, w: np.array, perceptrons: np.array):
        h = 0
        for i in range(len(w)):
            h += w[i] * perceptrons[i].o
        return h

    def calculate_delta_w(self, d, layer: np.array):
        d_w = []

        for i in range(len(layer)):
            d_w.append(self.eta * d * layer[i].o)
        return d_w

    def create_layer(self, q, w, is_last: bool):
        perceptrons = []
        for i in range(q):
            if i == q - 1 and not is_last:
                perceptrons.append(Perceptron(None))
            else:
                perceptrons.append(Perceptron(copy.deepcopy(w)))
        return np.array(perceptrons)

    def calculate_d(self, layer: np.array, idx):
        acum = 0

        for i in range(len(layer)):
            acum += layer[i].d * layer[i].w[idx]
        return acum


class Perceptron:

    def __init__(self, w: Optional['np.array']):
        self.w = w
        self.h = 0
        self.o = 0
        self.d = 0
