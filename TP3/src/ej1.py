import sys

import matplotlib.pyplot as plt
import numpy

from algorithms.Perceptron import SimplePerceptron, NoLinearPerceptron, LinearPerceptron
from utils.Config_p import Config
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

    perceptron_parameters: PerceptronParameters = PerceptronParameters(config)

    perceptron: SimplePerceptron
    if config.perceptron_algorithm == 'not_linear_perceptron':
        perceptron = NoLinearPerceptron(perceptron_parameters)
    elif config.perceptron_algorithm == 'linear_perceptron':
        y = 2 * (y - min(y)) / (max(y) - min(y)) - 1
        perceptron = LinearPerceptron(perceptron_parameters)
    else:
        perceptron = SimplePerceptron(perceptron_parameters)

    print('Running ' + config.perceptron_algorithm + '...')
    results = perceptron.train(x, y)
    print(config.perceptron_algorithm + ' finished.')

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
    plt.show()
    # plt.savefig(output_dir + '/' + config.selection_algorithm + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png')
    plt.clf()
    plt.figure(figsize=(7, 7), layout='constrained', dpi=200)
    x = range(results.iterations)
    y = results.errors
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Errores por iteración")
    plt.grid(True)
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    __main__()
