import sys

import numpy

from algorithms.Perceptron import perceptron
from utils.Config_p import Config
from utils.Functions import get_error, get_error_sign, sigmoide_logistic, sign, identity, sigmoide_tanh
from utils.PerceptronParameters import PerceptronParameters


def get_error_function(config: Config):
    if config.perceptron_algorithm == 'simple_perceptron':
        return get_error_sign
    else:
        return get_error


def get_activation_function(config: Config):
    if config.perceptron_algorithm == 'simple_perceptron':
        return sign
    elif config.perceptron_algorithm == 'no_linear_perceptron':
        if config.function == 'sigmoid_logistic':
            return sigmoide_logistic
        else:
            return sigmoide_tanh
    else:
        return identity


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 4, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config = Config(f.read())
    f.close()

    activation_function = get_activation_function(config)
    error_function = get_error_function(config)
    perceptron_parameters: PerceptronParameters = PerceptronParameters(config, activation_function, error_function)

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

    y: [] = []
    with open(sys.argv[3], 'r') as expected_outputs_file:
        for line in expected_outputs_file:
            v = float(line)
            if config.perceptron_algorithm == 'no_linear_perceptron':
                v = activation_function(v, perceptron_parameters)
            y.append(float(v))

    y = numpy.array(y)

    print('Running ' + config.perceptron_algorithm + '...')
    results = perceptron(perceptron_parameters, x, y)
    print(config.perceptron_algorithm + ' finished.')


if __name__ == "__main__":
    __main__()
