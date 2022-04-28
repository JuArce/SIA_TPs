import copy
from datetime import datetime

import math
import numpy
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

        self.w = []
        self.d_w = []

        # create w and d_w # TODO Inicializar el conjunto de pesos en valores ’pequeños’ al azar
        for i in range(len(self.layers) - 1):  # la capa de salida no tiene pesos
            aux = []
            for j in range(self.layers[i + 1]):  # recorro la cantidad de veces de la capa siguiente
                aux.append(np.zeros(self.layers[i] + 1))
            self.w.append(np.array(aux))
        self.w = np.array(self.w)
        self.d_w = np.array(self.w)

        # create h y o
        self.o = []
        for i in range(len(self.layers)):
            aux = np.zeros(self.layers[i] + 1) if i != len(self.layers) - 1 else np.zeros(self.layers[i])
            self.o.append(aux)

        self.o = np.array(self.o)

        self.errors = []

        # TODO: Terminar el vector de errores
        self.errors = []
        for i in range(self.len_layers, 1, - 1):  # recorro la cantidad de capas
            aux = np.zeros(self.layers[i - 1] + 1) if i != self.len_layers else np.zeros(self.layers[i - 1])
            self.errors.append(aux)

        self.errors = np.array(self.errors)

    def train(self):
        time = datetime.now()
        w_min = self.w
        error = 1
        h = []
        while error > self.max_error:
            idx = random.randint(0, len(self.x))

            self.o[0] = np.array(self.x[idx])
            # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
            for m in range(1, self.len_layers):  # por cada capa 1 a M
                aux = []
                for i in range(self.layers[m]):  # recorro todos los de la capa actual menos el del umbral
                    aux.append(self.w[m - 1][i] @ self.o[m - 1])
                if m != self.len_layers - 1:  # en la capa de salida no hay que agregar el valor de umbral
                    aux.append(self.o[m][len(self.o[m]) - 1])
                h.append(np.array(aux))
                self.o[m] = vectorize(pyfunc=self.activation_function)(aux)  # Estado de Activacion

            # Calculo d en la capa de salida
            h = np.array(h)
            aux = []
            for m in range(self.layers[self.len_layers - 1]):  # recorro la capa de salida
                aux.append(self.activation_function_derivative(h[self.len_layers - 2][m]) * (
                        self.y[idx] - self.o[self.len_layers - 1][m]))  # TODO: ver si recibe otro Yi que pasa

            self.errors[0] = numpy.array(aux)

            # Retropropagar hacia abajo

            for m in range(self.len_layers, 2, -1):  # recorro de M a 2
                actual_level = m - 2
                aux = []
                for i in range(self.layers[actual_level] + 1):  # tengo que calcular d del nivel actual
                    acum = 0
                    for j in range(self.layers[actual_level + 1]):  # tengo que recorrer todos los de la capa superior
                        acum += self.w[actual_level][i][j] * self.errors[len(self.errors) - 1][j]

                #         aux.append(self.activation_function_derivative(h[actual_level][i]) * acum)
                # errors[actual_level] = numpy.array(aux)

        return Results(self.x, self.y, self.w, self.algorithm, self.function, time)

    def predict(self):
        return None

    def activation_function(self, h):
        return h

    def activation_function_derivative(self, h):
        return 1
