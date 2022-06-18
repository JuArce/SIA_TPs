import math
from copy import deepcopy
from statistics import mean
from typing import Optional

import numpy as np
from numpy import random


class Network:
    LINEAR_FUNCTIONS = {
        "identity": {
            "f": lambda h: h,
            "fp": lambda h: 1
        },
    }

    NO_LINEAR_FUNCTIONS = {
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

    def __init__(self, config, layers):
        self.weights = self._build_weights(layers)
        if config.algorithm == 'no_linear_perceptron':
            self.act_function = Network.NO_LINEAR_FUNCTIONS[config.function]['f']
            self.act_function_derivative = Network.NO_LINEAR_FUNCTIONS[config.function]['fp']
        elif config.algorithm == 'linear_perceptron':
            self.act_function = Network.LINEAR_FUNCTIONS[config.function]['f']
            self.act_function_derivative = Network.LINEAR_FUNCTIONS[config.function]['fp']
        else:
            raise 'Invalid algorithm for multiperceptron'
        self.eta = config.eta
        self.cota = config.cota
        self.betha = config.betha
        self.algorithm = config.algorithm
        self.function = config.function
        self.max_error = config.max_error
        self.layers = layers
        self.w_by_layer = None

    def _build_weights(self, layers):
        weights = np.empty(len(layers) - 1, dtype=object)
        for i in range(len(weights)):
            weights[i] = np.random.uniform(-1, 1, size=(layers[i + 1], layers[i]))

        return weights

    def calculate_error(self, x, x_data, y_expected):

        errors = []
        weights = self.weights_resize(x)
        for i in range(len(x_data)):
            expected = y_expected[i]
            output = self._get_output(x_data[i], weights)
            errors.append((output - expected) ** 2)

        errors = np.array(errors)
        return mean((1 / len(x_data)) * sum(errors))

    def weights_resize(self, weights_array):

        weights = np.empty(len(self.layers) - 1, dtype=object)
        k = 0
        for i in range(len(weights)):
            weights[i] = np.empty((self.layers[i + 1], self.layers[i]))
            for j in range(len(weights[i])):
                size = len(weights[i][j])
                weights[i][j] = weights_array[k:k + len(weights[i][j])]
                k += size

        return weights

    def _get_output(self, input_x, weights):
        x = deepcopy(input_x)
        for i in range(len(weights)):
            h = []
            for j in range(len(weights[i])):
                h.append(x @ weights[i][j])
            x = list(map(self.activation_function, h))
        return x

    # [35, 2, 1, 2, 1, 2, 35]

    def activation_function(self, h):
        if self.algorithm == 'no_linear_perceptron':
            return self.act_function(h, self.betha)
        else:
            return self.act_function(h)

    def assign_weights(self, weights):
        self.weights = self.weights_resize(weights)


#

#

#
# def __init__(self, config, input_len, output_len):
#     if config.algorithm == 'no_linear_perceptron':
#         self.act_function = Network.NO_LINEAR_FUNCTIONS[config.function]['f']
#         self.act_function_derivative = Network.NO_LINEAR_FUNCTIONS[config.function]['fp']
#     elif config.algorithm == 'linear_perceptron':
#         self.act_function = Network.LINEAR_FUNCTIONS[config.function]['f']
#         self.act_function_derivative = Network.LINEAR_FUNCTIONS[config.function]['fp']
#     else:
#         raise 'Invalid algorithm for multiperceptron'
#     self.eta = config.eta
#     self.cota = config.cota
#     self.betha = config.betha
#     self.algorithm = config.algorithm
#     self.function = config.function
#     self.layers = self.build_layers_quantity(config.layers, input_len - 1, output_len)
#     self.max_error = config.max_error
#     self.w_by_layer = None
#
#     self.perceptrons: [[Perceptron]] = []
#
#     for i in range(len(self.layers)):
#         # si es el primero no le llega ningun peso
#         if i == 0:
#             self.perceptrons.append(self.create_layer(self.layers[i] + 1, i))
#
#         # si es el último tiene la cantidad de pesos de la capa inferior + 1 por el umbral
#         elif i == len(self.layers) - 1:
#
#             self.perceptrons.append(self.create_layer(self.layers[i], i))
#         else:
#             self.perceptrons.append(self.create_layer(self.layers[i] + 1, i))
#
# def train(self, x, y):
#     time = datetime.now()
#     error = 1
#     errors = []
#     std_devs = []
#
#     i = 0
#     while error > self.max_error and i < self.cota:
#         idx = random.randint(0, len(x))
#
#         # Propagar el estado de excitación y de activación a partir de  x[idx]
#         self.propagate(x, idx)
#
#         # calcular los estados de salida
#         self.calculate_d(y, idx)
#
#         # Calculando los nuevos pesos
#         self.calculate_delta_w()
#
#         # Calculo las funciones de activacion con todas las entradas
#         # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
#         output_activation = self.calculate_activation(x)
#         error = self.calculate_errors(x, y, output_activation)
#         errors.append(error)
#
#         if len(y[0]) > 1:
#             std_dev = self.calculate_std_dev(x, y, output_activation)
#             std_devs.append(std_dev)
#
#         i += 1
#
#     self.w_by_layer = self.build_w()
#     return Results(x, y, self.w_by_layer, self.algorithm, self.function,
#                    time, errors, self.max_error, i, std_devs=std_devs)
#
# def predict(self, x):
#     layers = []
#
#     for i in range(len(self.layers)):
#         layer = []
#         for n in range(len(self.perceptrons[i])):
#             perceptron = Perceptron(None, None, None, None)
#             if i != 0 and (n != len(self.perceptrons[i]) - 1 or i == len(self.layers) - 1):
#                 perceptron.o = 0
#                 perceptron.d = 0
#                 perceptron.h = 0
#             layer.append(perceptron)
#         layers.append(layer)
#
#     for i in range(len(self.perceptrons[0])):
#         layers[0][i].o = x[i]
#
#     for m in range(1, len(self.layers)):
#         for i in range(len(self.perceptrons[m])):
#             if i != len(self.perceptrons[m]) - 1 or m == len(self.layers) - 1:
#                 for j in range(len(self.perceptrons[m - 1])):
#                     layers[m][i].h += self.perceptrons[m][i].w[j] * layers[m - 1][j].o
#                 layers[m][i].o = self.activation_function(layers[m][i].h)
#             else:
#                 layers[m][i].o = 1
#
#     return list(map(lambda p: p.o, layers[-1]))
#
# def create_layer(self, q, layer_idx):
#     perceptrons = []
#     for i in range(q):
#         if layer_idx == 0 or (i == q - 1 and layer_idx != len(self.layers) - 1):
#             perceptrons.append(Perceptron(None, None, 1, None))
#         else:
#             perceptrons.append(Perceptron(random.uniform(-1, 1, size=self.layers[layer_idx - 1] + 1), 0, 0, 0))
#
#     return np.array(perceptrons)
#
# def propagate(self, x, idx):
#     # Le asigno a la capa de entrada los valores de entrada
#     for i in range(len(self.perceptrons[0])):
#         self.perceptrons[0][i].o = x[idx][i]
#
#     # Propago los estados de activación. Empiezo en 1 porque el 0 ya se calculo antes.
#     for m in range(1, len(self.layers)):  # por cada capa 1 a M
#         for i in range(len(self.perceptrons[m])):
#             self.perceptrons[m][i].h = 0
#
#             # recorro todos los de la capa actual menos el del umbral y que no sea la ultima capa
#             if i != len(self.perceptrons[m]) - 1 or m == len(self.layers) - 1:
#                 # Por capa neurona de la capa anterior
#                 for j in range(len(self.perceptrons[m - 1])):
#                     self.perceptrons[m][i].h += self.perceptrons[m][i].w[j] * self.perceptrons[m - 1][j].o
#                 self.perceptrons[m][i].o = self.activation_function(self.perceptrons[m][i].h)
#                 # Si es el umbral
#             else:
#                 self.perceptrons[m][i].o = 1
#
# def activation_function(self, h):
#     if self.algorithm == 'no_linear_perceptron':
#         return self.act_function(h, self.betha)
#     else:
#         return self.act_function(h)
#
# def activation_function_derivative(self, h):
#     if self.algorithm == 'no_linear_perceptron':
#         return self.act_function_derivative(h, self.betha)
#     else:
#         return self.act_function(h)
#
# def calculate_d(self, y, idx):
#     # Calculo d en la capa de salida
#     for i in range(len(self.perceptrons[-1])):  # recorro los perceptrones de la capa de salida
#         self.perceptrons[-1][i].d = self.activation_function_derivative(
#             self.perceptrons[-1][i].h) * \
#                                     (y[idx][i] - self.perceptrons[-1][i].o)
#     # Retropropagar el error
#     self.backpropagation()
#
# def backpropagation(self):
#
#     # Retropropagar hacia abajo
#     for m in range(len(self.layers) - 1, 1, -1):  # retropropagar de la capa de salida a la anteultima
#
#         for i in range(len(self.perceptrons[m - 1]) - 1):
#             aux = 0
#             # Por cada peso que sale de la neurona m-1
#             for j in range(len(self.perceptrons[m])):
#                 if j != len(self.perceptrons[m]) - 1 or m == len(self.layers) - 1:
#                     aux += self.perceptrons[m][j].w[i] * self.perceptrons[m][j].d
#             self.perceptrons[m - 1][i].d = self.activation_function_derivative(self.perceptrons[m - 1][i].h) * aux
#
# def calculate_delta_w(self):
#     for m in range(1, len(self.layers)):
#         # para cada neurona de la capa que estoy parado
#         for i in range(len(self.perceptrons[m])):
#             # Para cada neurona de la capa m-1
#             if i != (len(self.perceptrons[m])) - 1 or m == len(self.layers) - 1:
#                 for j in range(len(self.perceptrons[m - 1])):
#                     self.perceptrons[m][i].w[j] += self.eta * self.perceptrons[m][i].d * self.perceptrons[m - 1][
#                         j].o
#
# def calculate_delta_w_aux(self, d, layer: np.array):
#     d_w = []
#
#     for i in range(len(layer)):
#         d_w.append(self.eta * d * layer[i].o)
#     return np.array(d_w)
#
# def calculate_activation(self, x):
#     o = []
#     for i in range(len(x)):
#         o.append(self.predict(x[i]))
#     o = np.array(o)
#     return o
#
# def calculate_errors(self, x, y, o):
#     # Error cuadrático medio
#     return mean((1 / len(x)) * sum((y - o) ** 2))
#
# def calculate_std_dev(self, x, y, o):
#     return stdev((1 / len(x)) * sum((y - o) ** 2))
#
# def predict_set(self, x, y):
#     o = self.calculate_activation(x)
#     return self.calculate_errors(x, y, o)
#
# def predict_set_and_activation(self, x, y):
#     o = self.calculate_activation(x)
#     return self.calculate_errors(x, y, o), o
#
# def predict_set_with_multiple_outputs(self, x, y):
#
#     o = self.calculate_activation(x)
#     std_dev = None
#     if len(y[0]) > 0:
#         std_dev = self.calculate_std_dev(x, y, o)
#
#     return self.calculate_errors(x, y, o), std_dev
#
# def predict_set_with_multiple_outputs_and_activation(self, x, y):
#
#     o = self.calculate_activation(x)
#     std_dev = None
#     if len(y[0]) > 0:
#         std_dev = self.calculate_std_dev(x, y, o)
#
#     return self.calculate_errors(x, y, o), std_dev, o
#
# def build_w(self):
#     w = []
#
#     for m in range(1, len(self.layers)):
#         aux = []
#         for j in range(len(self.perceptrons[m])):
#             aux.append(self.perceptrons[m][j].w)
#         w.append(aux)
#     return w
#
# def build_layers_quantity(self, hidden_layers, input_layer_len, output_layer_len):
#     layers = [input_layer_len]
#     layers.extend(hidden_layers)
#     layers.append(output_layer_len)
#     return layers


class Perceptron:

    def __init__(self, w: Optional['np.array'], h: Optional['float'], o: Optional['float'], d: Optional['float']):
        self.w = w
        self.h = h
        self.o = o
        self.d = d
