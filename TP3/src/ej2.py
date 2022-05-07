import copy
import sys
import random
from datetime import datetime
from statistics import mean, stdev

import numpy
import numpy as np

from algorithms.Perceptron import SimplePerceptron, NoLinearPerceptron, LinearPerceptron
from utils.Config_p import Config
from utils.Graph import graph, graph_table
from utils.PerceptronParameters import PerceptronParameters
from utils.Utils import build_train


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
            # values = line.split()
            # aux = []
            # for v in values:
            #     aux.append(float(v))
            # y.append(aux)

            y.append(float(line))

    y = numpy.array(y)

    if config.perceptron_algorithm == 'not_linear_perceptron':
        y = 2 * (y - min(y)) / (max(y) - min(y)) - 1

    perceptron_parameters: PerceptronParameters = PerceptronParameters(config)

    perceptron: SimplePerceptron
    if config.perceptron_algorithm == 'not_linear_perceptron':
        y = 2 * (y - min(y)) / (max(y) - min(y)) - 1
        perceptron = NoLinearPerceptron(perceptron_parameters)
    elif config.perceptron_algorithm == 'linear_perceptron':
        perceptron = LinearPerceptron(perceptron_parameters)
    else:
        perceptron = SimplePerceptron(perceptron_parameters)

    print('Running ' + config.perceptron_algorithm + '...')

    # capacidad del perceptron para aprender la funciónn cuyas muestras están presentes
    # graficar todos los errores en el entrenamiento
    results = perceptron.train(x, y)
    output_dir = './errors_' + config.perceptron_algorithm + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    graph(range(results.iterations), results.errors, 'Iteración', 'Error', 'Errores por Iteración')

    # capacidad de generalización del perceptron
    indexes = [*range(len(x))]
    random.shuffle(indexes)
    indexes = np.array(indexes)
    indexes = np.array_split(indexes, k)

    # [[1 3 2], [9 5 2], ...[]]
    results_training = []
    results_test = []
    points = []
    colors = []
    cell_text = []
    train_errors = []
    test_errors = []
    train_stdev = []
    test_stdev = []

    columns = []
    for i in range(k):
        columns.append(str(i + 1))
    rows = ['Error Train', 'Std Dev Train', 'Error Test', 'Std Dev Test']

    for i in range(k):
        # reiniciamos el perceptron
        perceptron.__init__(perceptron_parameters)

        training_x, training_y, testing_x, testing_y = build_train(indexes, x, y, i)

        r_train = perceptron.train(training_x, training_y)
        r_test = perceptron.predict(testing_x, testing_y)

        # results_training.append(r_train)
        # results_test.append(r_test)

        points.append([i, r_train.errors[-1]])
        colors.append('#00ff3c')  # Green -> TRAIN

        points.append([i, r_test])
        colors.append('#fa0000')  # Red -> PREDICT

        train_errors.append(r_train.errors[-1])
        test_errors.append(r_test)
        train_stdev.append(get_3d_stddev(training_x))
        test_stdev.append(get_3d_stddev(testing_x))

    cell_text.append([round(n, 5) for n in train_errors])
    cell_text.append([round(n, 5) for n in train_stdev])
    cell_text.append([round(n, 5) for n in test_errors])
    cell_text.append([round(n, 5) for n in test_stdev])
    graph(x_label='k', y_label='error', title='Train (Green) vs Test (Red)', points=numpy.array(points),
          points_color=colors)
    graph_table(cell_text=cell_text, rows=rows, columns=columns)
    # normalizar la salida del no lineal

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


    # Seleccionar qué porcentaje es mejor tomar de población para entrenar y para testeo
    points = []
    errors = []
    colors = []

    percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for i in range(len(percentages)):
        i_train_errors = []
        i_test_errors = []
        for j in range(10):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            aux_x = x[indices]
            aux_y = y[indices]
            training_x = aux_x[0:int(len(aux_x) * percentages[i])]
            training_y = aux_y[0:int(len(aux_y) * percentages[i])]
            testing_x = aux_x[int(len(aux_x) * percentages[i]):]
            testing_y = aux_y[int(len(aux_y) * percentages[i]):]
            perceptron.__init__(perceptron_parameters)
            r_train = perceptron.train(training_x, training_y)
            r_test = perceptron.predict(testing_x, testing_y)

            i_train_errors.append(r_train.errors[-1])
            i_test_errors.append(r_test)

        points.append([percentages[i], mean(i_train_errors)])
        errors.append(stdev(i_train_errors))
        colors.append('#00ff3c')  # Green -> TRAIN

        points.append([percentages[i], mean(i_test_errors)])
        errors.append(stdev(i_test_errors))
        colors.append('#fa0000')  # Red -> PREDICT

    graph(points=numpy.array(points), points_color=colors, e=errors)

    print(config.perceptron_algorithm + ' finished.')


def train_aux(perceptron, x, y, betha, function, errors, parameters):
    parameters.betha = betha
    parameters.function = function
    perceptron.__init__(parameters)
    r_train = perceptron.train(x, y)
    errors.append(r_train.errors[-1])

    
def get_3d_stddev(x):
    x_axis_stdev = stdev(x[:, 0])
    y_axis_stdev = stdev(x[:, 1])
    z_axis_stdev = stdev(x[:, 2])
    return (x_axis_stdev + y_axis_stdev + z_axis_stdev) / 3



if __name__ == "__main__":
    __main__()
