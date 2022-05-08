import copy
import sys
from datetime import datetime

import numpy

from algorithms.Perceptron import MultiPerceptron
from utils.Config_p import Config
from utils.Graph import graph, graph_multi, graph_table
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

    print('Running ' + config.perceptron_algorithm + '...')
    results = perceptron.train(x, y)
    print(config.perceptron_algorithm + ' finished.')
    output_dir = './errors_' + config.perceptron_algorithm + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    graph(range(results.iterations), results.errors, 'x', 'y', 'Errores por Iteración', output_dir=output_dir)

    # Cuál es la mejor cantidad de capas y unidades
    layers = [[2], [2, 2], [3, 3], [6, 6], [3, 2, 3]]
    aux_parameters = copy.deepcopy(perceptron_parameters)

    x_graph = []
    y_graph = []
    labels = []
    for i in range(len(layers)):
        aux_parameters.layers = layers[i]
        perceptron.__init__(aux_parameters, len(x[0]), len(y[0]))
        results = perceptron.train(x, y)
        x_graph.append(range(results.iterations))
        y_graph.append(results.errors)
        labels.append(str(layers[i]))

    graph_multi(x_graph, y_graph, 'x', 'y', 'Errores por Iteración usando distinta cantidad de capas', labels)

    # Capacidad de generalización con los k
    indexes = get_shuffle_indexes(x, k)
    points = []
    colors = []
    for i in range(k):
        # reiniciamos el perceptron
        perceptron.__init__(perceptron_parameters, len(x[0]), len(y[0]))
        training_x, training_y, testing_x, testing_y = build_train(indexes, x, y, i)
        r_train = perceptron.train(training_x, training_y)
        r_test = perceptron.predict_set(testing_x, testing_y)
        points.append([i, r_test])
        colors.append('#fa0000')  # Red -> PREDICT
        points.append([i, r_train.errors[-1]])
        colors.append('#00ff3c')  # Green -> TRAIN
        # graph(range(r_train.iterations), r_train.errors, 'x', 'y', 'Errores por Iteración')

    graph(x_label='k', y_label='error', title='Train (Green) vs Test (Red)',
          points=numpy.array(points, dtype=object), points_color=colors)

    # Probabilidad de acertar
    indexes = get_shuffle_indexes(x, k)
    rows = []
    columns = ["Error de Entrenamiento", "Error de Test", "Acertó"]
    cell_text = []
    for i in range(k):
        perceptron.__init__(perceptron_parameters, len(x[0]), len(y[0]))
        training_x, training_y, testing_x, testing_y = build_train(indexes, x, y, i)
        r_train = perceptron.train(training_x, training_y)
        r_test, aux_o = perceptron.predict_set_and_activation(testing_x, testing_y)

        rows.append(i)
        hit = 'Yes' if (-0.5 < aux_o < 0.5 and testing_y[0][0] == 0) or (
                0.5 < aux_o < 1.5 and testing_y[0][0] == 1) else 'No'

        cell_text.append([round(r_train.errors[-1], 8), round(r_test, 8), hit])

    graph_table(cell_text=cell_text, rows=rows, columns=columns)


def train_aux(perceptron, x, y, betha, function, errors, parameters):
    parameters.betha = betha
    parameters.function = function
    perceptron.__init__(parameters, len(x[0]), len(y[0]))
    r_train = perceptron.train(x, y)
    errors.append(r_train.errors[-1])


if __name__ == "__main__":
    __main__()
