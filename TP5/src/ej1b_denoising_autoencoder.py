import sys

import numpy
import numpy as np
from numpy import mean, sum as npsum

from algorithms.Autoencoder import Autoencoder
from utils.fonts import font_2
from utils import SeaGraphV2
from utils.Config_A import Config_A
from utils.utils import mutate_pattern, to_bin_array, resize_letter


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 2, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config_A = Config_A(f.read())
    f.close()
    data = []
    letters_patterns = []
    letters = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']
    for letter in font_2:
        aux = to_bin_array(letter)
        data.append(np.concatenate(aux))
        letters_patterns.append(aux)

    letters_dict = dict(zip(letters, letters_patterns))

    autoencoder = Autoencoder(config, len(data[0]), config.layers, config.latent_code_len)

    # mutate all patterns and train
    # letters_patterns_mutated_to_train = mutate_pattern(data, 10)
    # autoencoder.train(letters_patterns_mutated_to_train, data)
    letters_patterns_mutated_to_train = []
    data_to_train = []
    for i in range(0, 10):
        letters_patterns_mutated_to_train.extend(mutate_pattern(data, 10))
        data_to_train.extend(data)
    autoencoder.train(letters_patterns_mutated_to_train, data_to_train)

    o = []
    for i in range(len(data)):
        o.append(autoencoder.get_output(letters_patterns_mutated_to_train[i]))
    o = numpy.array(o)
    error = mean((npsum((data - o) ** 2, axis=1) / 2))

    o_trained = []
    for i in range(len(data)):
        o_trained.append(autoencoder.get_output(letters_patterns_mutated_to_train[i]))
    o_trained = numpy.array(o_trained)
    error_trained = mean((npsum((data - o_trained) ** 2, axis=1) / 2))

    # generate a new set
    letters_patterns_mutated = mutate_pattern(data, 10)

    o_new_set = []
    for i in range(len(data)):
        o_new_set.append(autoencoder.get_output(letters_patterns_mutated[i]))
    o_new_set = numpy.array(o_new_set)
    error = mean((npsum((data - o_new_set) ** 2, axis=1) / 2))

    # resize all outputs
    letters_patterns_mutated_to_train = np.array(list(map(resize_letter, letters_patterns_mutated_to_train)))
    letters_patterns_mutated = np.array(list(map(resize_letter, letters_patterns_mutated)))
    o_trained = np.array(list(map(resize_letter, o_trained)))
    o_new_set = np.array(list(map(resize_letter, o_new_set)))
    o = np.array(list(map(resize_letter, o)))

    # Grafico de letras
    graphs = []
    for i, l in enumerate(letters[0:5]):
        graphs.append(letters_dict[l])  # letra original
        graphs.append(o[i])  # letra original obtenida
        graphs.append(letters_patterns_mutated_to_train[i])  # letra con ruido entrenada
        graphs.append(o_trained[i])  # salida obtenida con letra con ruido entrenada
        graphs.append(letters_patterns_mutated[i])  # letra con ruido no entrenada
        graphs.append(o_new_set[i])  # salida letra con ruido no entrenada
        if (i + 1) % 5 == 0:
            SeaGraphV2.graph_multi_heatmap(graphs, title='Letters', c_map="Greys", cols=6)
            graphs = []

    # Grafico de diferentes valores de ruido para la misma letra
    graphs = [letters_dict['A']]
    for noise in range(0, 16, 5):
        letters_patterns_mutated = mutate_pattern(data, noise)
        o_new_set = []
        for i in range(len(data)):
            o_new_set.append(autoencoder.get_output(letters_patterns_mutated[i]))
        o_new_set = numpy.array(o_new_set)
        graphs.extend(np.array(list(map(resize_letter, [o_new_set[0]]))))
    SeaGraphV2.graph_multi_heatmap(graphs, title='Letter A | Noise level {0, 5, 10, 15}', c_map="Greys", cols=6)


if __name__ == "__main__":
    __main__()
