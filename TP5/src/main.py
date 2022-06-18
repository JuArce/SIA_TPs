import sys

import numpy as np

from algorithms.Autoencoder import Autoencoder
from algorithms.fonts import font_2
from utils.Config_A import Config_A


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
    autoencoder.train(data, data)
    a = autoencoder.encode(data[0])
    b = autoencoder.decode(a)
    c = autoencoder.get_output(data[0])


def to_bin_array(encoded_caracter):
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(0, 7):
        current_row = encoded_caracter[row]
        for col in range(0, 5):
            bin_array[row][4 - col] = current_row & 1
            current_row >>= 1
    return bin_array  # bin_array = [[...],[...],...]


if __name__ == "__main__":
    __main__()
