import copy
import sys

import numpy

import random
import datetime
from algorithms.Perceptron import MultiPerceptron
from utils.Config_p import Config
from utils.Graph import graph, graph_table
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

    r_train = perceptron.train(x, y)
    graph(range(r_train.iterations), r_train.errors, 'Iteración', 'Error', 'Errores por Iteración')
    print("Last error: " + str(r_train.errors[-1]))
    print("Min error: " + str(min(r_train.errors)))

    points = []
    colors = []
    errors = []
    mut_prob = [0, 0.01, 0.1, 0.25, 0.5]
    rows = [str(i) for i in range(len(x))]
    columns = mut_prob
    cell_text = []

    # testing_x = mutate_input(x, 0.02)  # TODO: recibir por parámetro

    for i in range(len(x)):  # paso por cada numero
        # recorro cada probabilidad de mutacion
        aux = []
        for p in mut_prob:
            mut_x, changed = mutate_single_input(x[i], p)
            r_test_errors, r_test_std_devs, = perceptron.predict_set_with_multiple_outputs([mut_x], [y[i]])
            aux.append(str(round(r_test_errors, 8)) + " (" + str(changed) + ")")
        cell_text.append(aux)

    graph_table(cell_text=cell_text, rows=rows, columns=columns)


def mutate_input(x, mutation_prob):
    # random.seed(datetime.datetime.now())
    aux = copy.deepcopy(x)

    for i in range(len(aux)):
        for j in range(len(aux[i])):
            r = random.random()
            if r < mutation_prob:
                aux[i][j] = 1 if aux[i][j] == 0 else 0


def mutate_single_input(x, mutation_prob):
    # random.seed(datetime.datetime.now())
    aux = copy.deepcopy(x)
    changed = 0
    for i in range(len(aux)):
        r = random.random()
        if r < mutation_prob:
            aux[i] = 1 if aux[i] == 0 else 0
            changed += 1

    return aux, changed


if __name__ == "__main__":
    __main__()
