import sys
from datetime import datetime

import numpy

from algorithms.Perceptron import MultiPerceptron
from utils.Config_p import Config
from utils.Graph import graph
from utils.PerceptronParameters import PerceptronParameters


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
    print('Running ' + config.perceptron_algorithm + '...')
    results = perceptron.train(x, y)
    print(config.perceptron_algorithm + ' finished.')

    output_dir = './errors_' + config.perceptron_algorithm + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    graph(range(results.iterations), results.errors, 'x', 'y', 'Errores por Iteración', output_dir=output_dir)


if __name__ == "__main__":
    __main__()
