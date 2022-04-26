import sys

import matplotlib.pyplot as plt
import numpy

from algorithms.Perceptron import perceptron
from utils.Config_p import Config
from utils.Functions import get_error
from utils.PerceptronParameters import PerceptronParameters
from utils.activation_functions import sigmoide_logistic, sign, identity, sigmoide_tanh, sigmoide_logistic_derivative, \
    sigmoide_tanh_derivative
from utils.delta_functions import delta_function, delta_function_no_linear


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
    error_function = get_error

    if not config.perceptron_algorithm == 'no_linear_perceptron':
        activation_function_derivative = None
        delta_f = delta_function
    else:
        delta_f = delta_function_no_linear
        if config.function == 'sigmoid_logistic':
            activation_function_derivative = sigmoide_logistic_derivative
        else:
            activation_function_derivative = sigmoide_tanh_derivative

    perceptron_parameters: PerceptronParameters = PerceptronParameters(config, activation_function, error_function,
                                                                       delta_f, activation_function_derivative)

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
            y.append(float(line))

    y = numpy.array(y)

    if config.perceptron_algorithm == 'no_linear_perceptron':
        y = 2*(y - min(y))/(max(y)-min(y)) - 1

    print('Running ' + config.perceptron_algorithm + '...')
    results = perceptron(perceptron_parameters, x, y)
    print(config.perceptron_algorithm + ' finished.')

    for i, x in enumerate(results.x):
        e = (x @ results.w) - results.y[i]
        print(str(e))

    plt.figure(figsize=(7, 7), layout='constrained', dpi=200)
    plt.scatter(results.x[:, 0], results.x[:, 1], s=100, c=results.y)
    x = range(-2, 4)
    # -w_0/w_1 x - w_2/w_1
    y = (- (results.w[0] / results.w[1]) * x - (results.w[2] / results.w[1]))
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Separabilidad")
    plt.grid(True)
    # plt.legend()
    # plt.show()
    # plt.savefig(output_dir + '/' + config.selection_algorithm + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png')


if __name__ == "__main__":
    __main__()
