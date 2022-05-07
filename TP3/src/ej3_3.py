import copy
import sys

import numpy

import random
import datetime
from algorithms.Perceptron import MultiPerceptron
from utils.Config_p import Config
from utils.Graph import graph
from utils.PerceptronParameters import PerceptronParameters
from utils.Utils import build_train, get_shuffle_indexes


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 4, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config = Config(f.read())
    f.close()

    x = []
    number = []
    i = 1
    with open(sys.argv[2], 'r') as inputs_file:
        for line in inputs_file:
            values = line.split()
            aux = []
            for v in values:
                aux.append(float(v))

            number.extend(aux)

            if i % 7 == 0:
                number.append(1)
                x.append(number)
                number = []
            i += 1

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

    perceptron_parameters: PerceptronParameters = PerceptronParameters(config)
    perceptron: MultiPerceptron = MultiPerceptron(perceptron_parameters, len(x[0]), len(y[0]))

    points = []
    colors = []
    errors = []
    r_train = perceptron.train(x, y)

    testing_x = mutate_input(x, 0.02)  # TODO: recibir por parámetro

    for i in range(len(x)):
        r_test_errors, r_test_std_devs, = perceptron.predict_set_with_multiple_outputs([x[i]], [y[i]])
        r_test_errors_mut, r_test_std_devs_mut, = perceptron.predict_set_with_multiple_outputs([testing_x[i]], [y[i]])

        # points.append([i, r_test_errors])
        # errors.append(r_test_std_dev)
        # colors.append('#fa0000')  # Red -> PREDICT
        #
        # points.append([i, r_train.errors[-1]])
        # errors.append(r_train.std_devs[-1])
        # colors.append('#00ff3c')  # Green -> TRAIN
        # graph(range(r_train.iterations), r_train.errors, 'x', 'y', 'Errores por Iteración')

    graph(points=numpy.array(points), points_color=colors, e=errors)


def mutate_input(x, mutation_prob):
    random.seed(datetime.datetime.now())
    aux = copy.deepcopy(x)

    for i in range(len(aux)):
        for j in range(len(aux[i])):
            r = random.random()
            if r < mutation_prob:
                aux[i][j] = 1 if aux[i][j] == 0 else 0

    return aux


if __name__ == "__main__":
    __main__()
