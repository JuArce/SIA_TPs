import copy
from datetime import datetime
from typing import Optional

import math
import matplotlib.pyplot as plt
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
        self.betha = perceptron_parameters.betha
        self.algorithm = perceptron_parameters.algorithm
        self.function = perceptron_parameters.function
        self.layers = perceptron_parameters.layers
        self.max_error = perceptron_parameters.max_error
        self.len_layers = len(self.layers)

        self.perceptrons: [[Perceptron]] = []

        for i in range(self.len_layers):
            # si es el primero no le llega ningun peso
            if i == 0:
                self.perceptrons.append(self.create_layer(self.layers[i] + 1, i))

            # si es el último tiene la cantidad de pesos de la capa inferior + 1 por el umbral
            elif i == self.len_layers - 1:

                self.perceptrons.append(self.create_layer(self.layers[i], i))
            else:
                self.perceptrons.append(self.create_layer(self.layers[i] + 1, i))

    def train(self):
        time = datetime.now()
        error = 1
        errors = []
        i = 0
        while error > self.max_error and i < 50000:
            idx = random.randint(0, len(self.x))

            # Propagar el estado de excitación y de activación a partir de  x[idx]
            self.propagate(idx)

            # calcular los estados de salida
            self.calculate_d(idx)

            # Calculando los nuevos pesos
            self.calculate_delta_w()

            # Calculo las funciones de activacion con todas las entradas
            # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
            error = self.calculate_errors()
            errors.append(error)

            i += 1
        plt.figure(dpi=200)
        plt.plot([*range(len(errors))], errors)
        plt.show()

        o = self.predict()
        y = self.y

        return Results(self.x, self.y, self.w, self.algorithm, self.function, time)

    def predict(self):
        o = []
        perceptrons = copy.deepcopy(self.perceptrons)
        for j in range(len(self.x)):
            for i in range(len(perceptrons[0])):
                perceptrons[0][i].o = self.x[j][i]
                perceptrons[0][i].h = None

            # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
            for m in range(1, self.len_layers):  # por cada capa 1 a M
                for i in range(self.layers[m]):  # recorro todos los de la capa actual menos el del umbral
                    perceptrons[m][i].h = self.calculate_h(perceptrons[m][i].w, perceptrons[
                        m - 1])  # le paso los pesos que me llegan y la capa anterior
                    perceptrons[m][i].o = self.activation_function(perceptrons[m][i].h)

            aux = []
            for i in range((self.layers[-1])):
                aux.append(perceptrons[-1][i].o)
            o.append(aux)
        return o

    def create_layer(self, q, layer_idx):
        perceptrons = []
        for i in range(q):
            if layer_idx == 0 or (i == q - 1 and layer_idx != len(self.layers) - 1):
                perceptrons.append(Perceptron(None, None, 1, None))
            else:
                perceptrons.append(Perceptron(random.uniform(-1, 1, size=self.layers[layer_idx - 1] + 1), 0, 0, 0))

        return np.array(perceptrons)

    def propagate(self, idx):
        # Le asigno a la capa de entrada los valores de entrada
        for i in range(len(self.perceptrons[0])):
            self.perceptrons[0][i].o = self.x[idx][i]

        # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
        for m in range(1, self.len_layers):  # por cada capa 1 a M
            for i in range(self.layers[m]):
                # recorro todos los de la capa actual menos el del umbral
                self.perceptrons[m][i].h = self.calculate_h(self.perceptrons[m][i].w, self.perceptrons[
                    m - 1])  # le paso los pesos que me llegan y la capa anterior
                self.perceptrons[m][i].o = self.activation_function(self.perceptrons[m][i].h)

    def calculate_h(self, w: np.array, perceptrons: np.array):
        h = 0
        for i in range(len(w)):
            h += w[i] * perceptrons[i].o
        return h

    def activation_function(self, h):
        return math.tanh(h * self.betha)

    def activation_function_derivative(self, h):
        return self.betha * (1 - (math.tanh(h * self.betha) ** 2))

    def calculate_d(self, idx):
        # Calculo d en la capa de salida

        for i in range(len(self.perceptrons[self.len_layers - 1])):  # recorro los perceptrones de la capa de salida
            self.perceptrons[self.len_layers - 1][i].d = self.activation_function_derivative(
                self.perceptrons[self.len_layers - 1][i].h) * \
                                                         (self.y[idx][i] - self.perceptrons[self.len_layers - 1][i].o)
        # Retropropagar el error
        self.backpropagation()

    def backpropagation(self):

        # Retropropagar hacia abajo

        for m in range(self.len_layers - 1, 1, -1):  # retropropagar de la capa de salida a la anteultima
            for i in range(len(self.perceptrons[m - 1]) - 1):
                h = self.perceptrons[m - 1][i].h
                self.perceptrons[m - 1][i].d = self.activation_function_derivative(h) * self.calculate_d_aux(
                    self.perceptrons[m], i, m)

    def calculate_d_aux(self, layer: np.array, idx, layer_number):
        acum = 0
        q = len(layer) if layer_number == self.len_layers - 1 else len(layer) - 1

        for i in range(q):
            acum += layer[i].d * layer[i].w[idx]
        return acum

    def calculate_delta_w(self):
        for m in range(self.len_layers - 1, 0, -1):
            q = len(self.perceptrons[m]) if m == self.len_layers - 1 else len(self.perceptrons[m]) - 1
            for i in range(q):
                self.perceptrons[m][i].w = self.perceptrons[m][i].w + self.calculate_delta_w_aux(
                    self.perceptrons[m][i].d, self.perceptrons[m - 1])

    def calculate_delta_w_aux(self, d, layer: np.array):
        d_w = []

        for i in range(len(layer)):
            d_w.append(self.eta * d * layer[i].o)
        return np.array(d_w)

    def calculate_errors(self):
        o = []
        perceptrons = copy.deepcopy(self.perceptrons)
        for i in range(len(self.x)):
            for j in range(len(perceptrons[0])):
                perceptrons[0][j].o = self.x[i][j]

            # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
            for m in range(1, self.len_layers):  # por cada capa 1 a M
                for j in range(self.layers[m]):  # recorro todos los de la capa actual menos el del umbral
                    perceptrons[m][j].h = self.calculate_h(perceptrons[m][j].w, perceptrons[
                        m - 1])  # le paso los pesos que me llegan y la capa anterior
                    perceptrons[m][j].o = self.activation_function(perceptrons[m][j].h)

            aux = []
            for j in range((self.layers[-1])):
                aux.append(perceptrons[-1][j].o)
            o.append(aux)

        return self.error_function(self.y, np.array(o))

    def error_function(self, y: np.ndarray, o: np.ndarray):
        return ((sum(sum(y - o))) ** 2) / 2


class Perceptron:

    def __init__(self, w: Optional['np.array'], h: Optional['float'], o: Optional['float'], d: Optional['float']):
        self.w = w
        self.h = h
        self.o = o
        self.d = d
