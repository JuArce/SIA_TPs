import sys

import numpy

from algorithms.Perceptron import MultiPerceptron
from utils.Config_p import Config
from utils.Graph import graph
from utils.PerceptronParameters import PerceptronParameters
from utils.Utils import build_train, get_shuffle_indexes


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 3, 'Missing arguments'
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

    y: [] = []  # 0 si es par, 1 si es impar.
    for i in range(10):
        aux = [i % 2]
        y.append(numpy.array(aux))

    y = numpy.array(y)

    perceptron_parameters: PerceptronParameters = PerceptronParameters(config)
    perceptron: MultiPerceptron = MultiPerceptron(perceptron_parameters, len(x[0]), len(y[0]))

    results_training = []
    results_test = []
    indexes = get_shuffle_indexes(x, k)
    points = []
    colors = []
    for i in range(k):
        training_x, training_y, testing_x, testing_y = build_train(indexes, x, y, i)
        r_train = perceptron.train(training_x, training_y)
        r_test = perceptron.predict_set(testing_x, testing_y)
        points.append([i, r_test])
        colors.append('#fa0000')  # Red -> PREDICT
        points.append([i, r_train.errors[-1]])
        colors.append('#00ff3c')  # Green -> TRAIN
        # graph(range(r_train.iterations), r_train.errors, 'x', 'y', 'Errores por Iteración')

    graph(points=numpy.array(points, dtype=object), points_color=colors)


if __name__ == "__main__":
    __main__()
