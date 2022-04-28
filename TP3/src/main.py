import sys

import matplotlib.pyplot as plt
import numpy

from algorithms.Perceptron import SimplePerceptron, NoLinearPerceptron, LinearPerceptron, MultiPerceptron
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
            y.append(float(line))

    y = numpy.array(y)

    perceptron_parameters: PerceptronParameters = PerceptronParameters(config)
    perceptron: MultiPerceptron = MultiPerceptron(x, y, perceptron_parameters)
    perceptron.train()
    if len(x) > 0:
        return 0

    perceptron: SimplePerceptron

    if config.perceptron_algorithm == 'no_linear_perceptron':
        perceptron = NoLinearPerceptron(x, y, perceptron_parameters)
    elif config.perceptron_algorithm == 'linear_perceptron':
        perceptron = LinearPerceptron(x, y, perceptron_parameters)
    else:
        perceptron = SimplePerceptron(x, y, perceptron_parameters)

    print('Running ' + config.perceptron_algorithm + '...')
    results = perceptron.train_perceptron()
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
