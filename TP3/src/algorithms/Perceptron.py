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

    def __init__(self, perceptron_parameters: PerceptronParameters):
        self.eta = perceptron_parameters.eta
        self.cota = perceptron_parameters.cota
        self.algorithm = perceptron_parameters.algorithm
        self.function = perceptron_parameters.function
        self.w = None

    def activation_function(self, h):
        if h >= 0:
            return 1
        return -1

    def delta_function(self, x: np.ndarray, y: np.ndarray, h: np.ndarray, o: np.ndarray, idx: int):
        return self.eta * (y[idx] - o[idx]) * x[idx]

    def error_function(self, y: np.ndarray, o: np.ndarray):
        return sum((y - o) ** 2) / 2

    def train(self, x, y):
        time = datetime.now()
        i = 0
        w = np.zeros(len(x[0]))
        w_min = w
        error = 1
        error_min = 2 * len(x)
        errors = []

        while error > 0 and i < self.cota:

            idx = random.randint(0, len(x))
            h: ndarray = x @ w  # producto interno (válida desde python 3.5) Estado de excitacion
            o: ndarray = vectorize(pyfunc=self.activation_function)(h)  # Estado de Activacion
            delta_w = self.delta_function(x, y, h, o, idx)
            w = w + delta_w
            error = self.error_function(y, o)

            if error < error_min:
                error_min = error
                w_min = copy.deepcopy(w)
            i += 1
            self.w = w
            errors.append(self.predict(x, y))

        self.w = w_min

        return Results(x, y, self.w, self.algorithm, self.function, time, errors, 0, i)

    def predict(self, x: np.ndarray, y: np.ndarray):

        h: ndarray = x @ self.w
        o: ndarray = vectorize(pyfunc=self.activation_function)(h)
        error = self.error_function(y, o)

        return error


class LinearPerceptron(SimplePerceptron):
    FUNCTIONS = {
        "identity": {
            "f": lambda h: h,
            "fp": lambda h: 1
        },
    }

    def __init__(self, perceptron_parameters: PerceptronParameters):
        super().__init__(perceptron_parameters)
        self.act_function = LinearPerceptron.FUNCTIONS[perceptron_parameters.function]["f"]
        self.act_function_derivative = LinearPerceptron.FUNCTIONS[perceptron_parameters.function]["fp"]

    def activation_function(self, h):
        return self.act_function(h)


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

    def __init__(self, perceptron_parameters: PerceptronParameters):
        super().__init__(perceptron_parameters)
        self.betha = perceptron_parameters.betha
        self.act_function = NoLinearPerceptron.FUNCTIONS[perceptron_parameters.function]["f"]
        self.act_function_derivative = NoLinearPerceptron.FUNCTIONS[perceptron_parameters.function]["fp"]

    def activation_function(self, h):
        return self.act_function(h, self.betha)

    def delta_function(self, x: np.ndarray, y: np.ndarray, h: np.ndarray, o: np.ndarray, idx: int):
        return self.eta * (y[idx] - o[idx]) * x[idx] * self.act_function_derivative(h[idx], self.betha)


class MultiPerceptron:

    def __init__(self, perceptron_parameters: PerceptronParameters):
        if perceptron_parameters.algorithm == 'no_linear_perceptron':
            self.act_function = NoLinearPerceptron.FUNCTIONS[perceptron_parameters.function]['f']
            self.act_function_derivative = NoLinearPerceptron.FUNCTIONS[perceptron_parameters.function]['fp']
        elif perceptron_parameters.algorithm == 'linear_perceptron':
            self.act_function = LinearPerceptron.FUNCTIONS[perceptron_parameters.function]['f']
            self.act_function_derivative = LinearPerceptron.FUNCTIONS[perceptron_parameters.function]['fp']
        else:
            raise 'Invalid algorithm for multiperceptron '
        self.eta = perceptron_parameters.eta
        self.cota = perceptron_parameters.cota
        self.betha = perceptron_parameters.betha
        self.algorithm = perceptron_parameters.algorithm
        self.function = perceptron_parameters.function
        self.layers = perceptron_parameters.layers
        self.max_error = perceptron_parameters.max_error

        self.perceptrons: [[Perceptron]] = []

        for i in range(len(self.layers)):
            # si es el primero no le llega ningun peso
            if i == 0:
                self.perceptrons.append(self.create_layer(self.layers[i] + 1, i))

            # si es el último tiene la cantidad de pesos de la capa inferior + 1 por el umbral
            elif i == len(self.layers) - 1:

                self.perceptrons.append(self.create_layer(self.layers[i], i))
            else:
                self.perceptrons.append(self.create_layer(self.layers[i] + 1, i))

    def train(self, x, y):
        time = datetime.now()
        error = 1
        errors = []
        i = 0
        while error > self.max_error and i < 50000:
            idx = random.randint(0, len(x))

            # Propagar el estado de excitación y de activación a partir de  x[idx]
            self.propagate(x, idx)

            # calcular los estados de salida
            self.calculate_d(y, idx)

            # Calculando los nuevos pesos
            self.calculate_delta_w()

            # Calculo las funciones de activacion con todas las entradas
            # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
            error = self.calculate_errors(x, y)
            errors.append(error)

            i += 1

        plt.figure(dpi=200)
        plt.plot([*range(len(errors))], errors)
        plt.show()

        return Results(x, y, self.build_w(), self.algorithm, self.function,
                       time, errors, self.max_error, i)

    def predict(self, x):
        layers = []

        for i in range(len(self.layers)):
            layer = []
            for n in range(len(self.perceptrons[i])):
                perceptron = Perceptron(None, None, None, None)
                if i != 0 and (n != len(self.perceptrons[i]) - 1 or i == len(self.layers) - 1):
                    perceptron.o = 0
                    perceptron.d = 0
                    perceptron.h = 0
                layer.append(perceptron)
            layers.append(layer)

        for i in range(len(self.perceptrons[0])):
            layers[0][i].o = x[i]

        for m in range(1, len(self.layers)):
            for i in range(len(self.perceptrons[m])):
                if i != len(self.perceptrons[m]) - 1 or m == len(self.layers) - 1:
                    for j in range(len(self.perceptrons[m - 1])):
                        layers[m][i].h += self.perceptrons[m][i].w[j] * layers[m - 1][j].o
                    layers[m][i].o = self.activation_function(layers[m][i].h)
                else:
                    layers[m][i].o = 1

        return list(map(lambda p: p.o, layers[-1]))

    def create_layer(self, q, layer_idx):
        perceptrons = []
        for i in range(q):
            if layer_idx == 0 or (i == q - 1 and layer_idx != len(self.layers) - 1):
                perceptrons.append(Perceptron(None, None, 1, None))
            else:
                perceptrons.append(Perceptron(random.uniform(-1, 1, size=self.layers[layer_idx - 1] + 1), 0, 0, 0))

        return np.array(perceptrons)

    def propagate(self, x, idx):
        # Le asigno a la capa de entrada los valores de entrada
        for i in range(len(self.perceptrons[0])):
            self.perceptrons[0][i].o = x[idx][i]

        # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
        for m in range(1, len(self.layers)):  # por cada capa 1 a M
            for i in range(len(self.perceptrons[m])):
                self.perceptrons[m][i].h = 0

                # recorro todos los de la capa actual menos el del umbral y que no sea la ultima capa
                if i != len(self.perceptrons[m]) - 1 or m == len(self.layers) - 1:
                    # Por capa neurona de la capa anterior
                    for j in range(len(self.perceptrons[m - 1])):
                        self.perceptrons[m][i].h += self.perceptrons[m][i].w[j] * self.perceptrons[m - 1][j].o
                    self.perceptrons[m][i].o = self.activation_function(self.perceptrons[m][i].h)
                    # Si es el umbral
                else:
                    self.perceptrons[m][i].o = 1

    def activation_function(self, h):
        if self.algorithm == 'no_linear_perceptron':
            return self.act_function(h, self.betha)
        else:
            return self.act_function(h)

    def activation_function_derivative(self, h):
        if self.algorithm == 'no_linear_perceptron':
            return self.act_function_derivative(h, self.betha)
        else:
            return self.act_function(h)

    def calculate_d(self, y, idx):
        # Calculo d en la capa de salida
        for i in range(len(self.perceptrons[-1])):  # recorro los perceptrones de la capa de salida
            self.perceptrons[-1][i].d = self.activation_function_derivative(
                self.perceptrons[-1][i].h) * \
                                        (y[idx][i] - self.perceptrons[-1][i].o)
        # Retropropagar el error
        self.backpropagation()

    def backpropagation(self):

        # Retropropagar hacia abajo
        for m in range(len(self.layers) - 1, 1, -1):  # retropropagar de la capa de salida a la anteultima

            for i in range(len(self.perceptrons[m - 1]) - 1):
                aux = 0
                # Por cada peso que sale de la neurona m-1
                for j in range(len(self.perceptrons[m])):
                    if j != len(self.perceptrons[m]) - 1 or m == len(self.layers) - 1:
                        aux += self.perceptrons[m][j].w[i] * self.perceptrons[m][j].d
                self.perceptrons[m - 1][i].d = self.activation_function_derivative(self.perceptrons[m - 1][i].h) * aux

    def calculate_delta_w(self):
        for m in range(1, len(self.layers)):
            # para cada neurona de la capa que estoy parado
            for i in range(len(self.perceptrons[m])):
                # Para cada neurona de la capa m-1
                if i != (len(self.perceptrons[m])) - 1 or m == len(self.layers) - 1:
                    for j in range(len(self.perceptrons[m - 1])):
                        self.perceptrons[m][i].w[j] += self.eta * self.perceptrons[m][i].d * self.perceptrons[m - 1][
                            j].o

    def calculate_delta_w_aux(self, d, layer: np.array):
        d_w = []

        for i in range(len(layer)):
            d_w.append(self.eta * d * layer[i].o)
        return np.array(d_w)

    def calculate_errors(self, x, y):
        o = []
        for i in range(len(x)):
            o.append(self.predict(x[i]))
        o = np.array(o)
        return 0.5 * sum((y - o) ** 2)

    def error_function(self, y: np.ndarray, o: np.ndarray):
        return ((sum(sum(y - o))) ** 2) / 2

    def build_w(self):

        w = []

        for m in range(1, len(self.layers)):
            aux = []
            for j in range(len(self.perceptrons[m])):
                aux.append(self.perceptrons[m][j].w)
            w.append(aux)
        return w


class Perceptron:

    def __init__(self, w: Optional['np.array'], h: Optional['float'], o: Optional['float'], d: Optional['float']):
        self.w = w
        self.h = h
        self.o = o
        self.d = d
