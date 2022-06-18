import sys

import numpy
import numpy as np
from numpy import mean, sum as npsum

from algorithms.Autoencoder import Autoencoder
from algorithms.fonts import font_2
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
    letters_patterns_mutated_to_train = []
    for i in range(len(data)):
        letters_patterns_mutated_to_train.append(mutate_pattern(data[i], 10))

    autoencoder.train(letters_patterns_mutated_to_train, data)

    # generate a new set
    letters_patterns_mutated = []
    for i in range(len(data)):
        letters_patterns_mutated.append(mutate_pattern(data[i], 10))

    o = []
    for i in range(len(data)):
        o.append(autoencoder.get_output(letters_patterns_mutated[i]))
    o = numpy.array(o)
    error = mean((npsum((data - o) ** 2, axis=1) / 2))

    # resize all outputs
    o = np.array(list(map(resize_letter, o)))

    # Grafico de letras


if __name__ == "__main__":
    __main__()
