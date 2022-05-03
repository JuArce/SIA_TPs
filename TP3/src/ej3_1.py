import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy
import numpy as np

from algorithms.Perceptron import MultiPerceptron
from utils.Config_p import Config
from utils.Graph import graph
from utils.PerceptronParameters import PerceptronParameters


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
            values = line.split()
            aux = []
            for v in values:
                aux.append(float(v))
            y.append(numpy.array(aux))

    y = numpy.array(y)

    if config.perceptron_algorithm == 'not_linear_perceptron':
        y = 2 * (y - min(y)) / (max(y) - min(y)) - 1

    perceptron_parameters: PerceptronParameters = PerceptronParameters(config)

    perceptron: MultiPerceptron = MultiPerceptron(perceptron_parameters)
    print('Running ' + config.perceptron_algorithm + '...')
    results = perceptron.train(x, y)
    print(config.perceptron_algorithm + ' finished.')

    output_dir = './errors_' + config.perceptron_algorithm + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    graph(range(results.iterations), results.errors, 'x', 'y', 'Errores por Iteración', output_dir=output_dir)


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
