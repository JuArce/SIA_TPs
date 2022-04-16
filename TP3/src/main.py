import numpy

from algorithms.SimplePerceptron import simple_perceptron
from utils.PerceptronParameters import PerceptronParameters
from utils.Config_p import Config
import sys

algorithms = {
    "simple_perceptron": simple_perceptron,
    "linear_perceptron": None,
    "not_linear_perceptron": None,
    "multi_layer_perceptron": None
}

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
print(x)

y: [] = []
with open(sys.argv[3], 'r') as expected_outputs_file:
    for line in expected_outputs_file:
        y.append(float(line))
y = numpy.array(y)
print(y)

perceptron_parameters: PerceptronParameters = PerceptronParameters(config)

print('Running ' + config.perceptron_algorithm + '...')
results = algorithms[config.perceptron_algorithm](perceptron_parameters, x, y)
print(config.perceptron_algorithm + ' finished.')
