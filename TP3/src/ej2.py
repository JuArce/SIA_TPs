import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy
import numpy as np

from algorithms.Perceptron import SimplePerceptron, NoLinearPerceptron, LinearPerceptron
from utils.Config_p import Config
from utils.PerceptronParameters import PerceptronParameters
from utils.Graph import graph, graph_init, graph_plot


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 4, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config = Config(f.read())
    f.close()

    x = []
    with open(sys.argv[2], 'r') as inputs_file:
        for line in inputs_file:
            values = line.split()
            aux = []
            for v in values:
                aux.append(float(v))
            aux.append(float(1))
            x.append(aux)
    x = numpy.array(x)

    k = config.k
    if len(x) % k != 0:
        print("length of training set is not divisible by k-fold parameter.")
        return 0

    y: [] = []
    with open(sys.argv[3], 'r') as expected_outputs_file:
        for line in expected_outputs_file:
            # values = line.split()
            # aux = []
            # for v in values:
            #     aux.append(float(v))
            # y.append(aux)

            y.append(float(line))

    y = numpy.array(y)

    if config.perceptron_algorithm == 'not_linear_perceptron':
        y = 2 * (y - min(y)) / (max(y) - min(y)) - 1

    perceptron_parameters: PerceptronParameters = PerceptronParameters(config)

    perceptron: SimplePerceptron
    if config.perceptron_algorithm == 'not_linear_perceptron':
        y = 2 * (y - min(y)) / (max(y) - min(y)) - 1
        perceptron = NoLinearPerceptron(perceptron_parameters)
    elif config.perceptron_algorithm == 'linear_perceptron':
        perceptron = LinearPerceptron(perceptron_parameters)
    else:
        perceptron = SimplePerceptron(perceptron_parameters)

    print('Running ' + config.perceptron_algorithm + '...')

    # capacidad del perceptron para aprender la funciónn cuyas muestras están presentes
    # graficar todos los errores en el entrenamiento
    results = perceptron.train(x, y)
    output_dir = './errors_' + config.perceptron_algorithm + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    graph(range(results.iterations), results.errors, 'x', 'y', 'Errores por Iteración')
    #reiniciamos el perceptron
    perceptron.__init__(perceptron_parameters)

    # capacidad de generalización del perceptron
    indexes = [*range(len(x))]
    random.shuffle(indexes)
    indexes = np.array(indexes)
    indexes = np.array_split(indexes, k)

    # [[1 3 2], [9 5 2], ...[]]
    results_training = []
    results_test = []
    points = []
    colors = []
    for i in range(k):
        training_x, training_y, testing_x, testing_y = build_train(indexes, x, y, i)
        r_train = perceptron.train(training_x, training_y)
        r_test = perceptron.predict(testing_x, testing_y)
        results_training.append(r_train)
        results_test.append(r_test)
        points.append([i, r_test])
        colors.append('#fa0000')  # Red -> PREDICT
        points.append([i, r_train.errors[-1]])
        colors.append('#00ff3c')  # Green -> TRAIN

    graph(points=numpy.array(points), points_color=colors)
    # normalizar la salida del no lineal



    # results = perceptron.train(training_x, training_y)

    print(config.perceptron_algorithm + ' finished.')


def build_train(indexes: np.array, data_x: np.array, data_y: np.array, idx: int):
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    for i in range(len(indexes)):
        for j in indexes[i]:
            if i != idx:
                test_set_x.append(data_x[j])
                test_set_y.append(data_y[j])
            else:
                train_set_x.append(data_x[j])
                train_set_y.append(data_y[j])

    return np.array(train_set_x), np.array(train_set_y), np.array(test_set_x), np.array(test_set_y)


def build_test():
    return None


if __name__ == "__main__":
    __main__()
