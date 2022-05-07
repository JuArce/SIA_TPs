import copy
import sys
from datetime import datetime

import numpy

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

    perceptron_parameters: PerceptronParameters = PerceptronParameters(config)

    perceptron: MultiPerceptron = MultiPerceptron(perceptron_parameters, len(x[0]), len(y[0]))

    # Seleccionar cuál es el mejor betha para entrenar a la red
    bethas = [0.1, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 2]
    aux_parameters = copy.deepcopy(perceptron_parameters)
    errors_logistic = []
    errors_tanh = []

    for i in range(len(bethas)):
        train_aux(perceptron, x, y, bethas[i], 'tanh', errors_tanh, aux_parameters)
        train_aux(perceptron, x, y, bethas[i], 'logistic', errors_logistic, aux_parameters)

    graph(bethas, errors_logistic, 'Betha', 'Error', 'Errores para distintos bethas (logistic)')
    graph(bethas, errors_tanh, 'Betha', 'Error', 'Errores para distintos bethas (tanh)')

    print('Running ' + config.perceptron_algorithm + '...')
    results = perceptron.train(x, y)
    print(config.perceptron_algorithm + ' finished.')

    output_dir = './errors_' + config.perceptron_algorithm + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    graph(range(results.iterations), results.errors, 'x', 'y', 'Errores por Iteración', output_dir=output_dir)


def train_aux(perceptron, x, y, betha, function, errors, parameters):
    parameters.betha = betha
    parameters.function = function
    perceptron.__init__(parameters, len(x[0]), len(y[0]))
    r_train = perceptron.train(x, y)
    errors.append(r_train.errors[-1])


if __name__ == "__main__":
    __main__()
