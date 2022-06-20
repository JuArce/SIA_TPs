import sys

import numpy
import numpy as np
from numpy import mean, sum as npsum

from algorithms.Autoencoder import Autoencoder
from utils import SeaGraphV2
from utils.Config_A import Config_A
from utils.fonts import font_2
from utils.utils import to_bin_array, resize_letter, midpoint


def generate_new_letters(letters_dict, autoencoder, letters):
    graphs = []
    indexs = np.random.choice(len(letters), 2, replace=False)

    letter_1 = letters[indexs[0]]
    letter_2 = letters[indexs[1]]

    letter_1_array = np.concatenate(letters_dict[letter_1])
    letter_2_array = np.concatenate(letters_dict[letter_2])
    letter_1_encode = autoencoder.encode(letter_1_array)
    letter_2_encode = autoencoder.encode(letter_2_array)

    direction = letter_2_encode - letter_1_encode  # de 1 a 2

    parts = 4

    proportion = direction / parts

    graphs.append(letters_dict[letter_1])

    letter_1_decode = autoencoder.decode(letter_1_encode)
    res = np.array(list(map(resize_letter, [letter_1_decode])))
    graphs.append(res[0])

    for i in range(1, parts):
        letter_3_encode = letter_1_encode + proportion * i
        letter_3_decode = autoencoder.decode(letter_3_encode)
        res = np.array(list(map(resize_letter, [letter_3_decode])))
        graphs.append(res[0])

    letter_3_encode = midpoint(letter_1_encode[0], letter_1_encode[1], letter_2_encode[0], letter_2_encode[1])
    letter_3_decode = autoencoder.decode(letter_3_encode)
    res = np.array(list(map(resize_letter, [letter_3_decode])))
    graphs.append(res[0])

    letter_2_decode = autoencoder.decode(letter_2_encode)
    res = np.array(list(map(resize_letter, [letter_2_decode])))
    graphs.append(res[0])

    graphs.append(letters_dict[letter_2])

    SeaGraphV2.graph_multi_heatmap(graphs, title='New Letter', c_map="Greys", cols=4)


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

    o_trained = []
    for i in range(len(data)):
        o_trained.append(autoencoder.get_output(data[i]))
    o_trained = numpy.array(o_trained)
    error_trained = mean((npsum((data - o_trained) ** 2, axis=1) / 2))
    print("Train error: " + str(error_trained))

    x = []
    y = []
    for key, value in letters_dict.items():
        res = autoencoder.encode(np.concatenate(value))
        x.append(res[0])
        y.append(res[1])
    SeaGraphV2.graph_points(x, y, list(letters_dict.keys()), title="Capa Latente")

    graphs = []
    for i, l in enumerate(letters):
        graphs.append(letters_dict[l])  # letra original
        res = autoencoder.get_output(np.concatenate(letters_dict[l]))
        res = np.array(res)
        res = np.array(list(map(resize_letter, [res])))
        graphs.append(res[0])
        if (i + 1) % 8 == 0:
            SeaGraphV2.graph_multi_heatmap(graphs, title='Letters', c_map="Greys", cols=4)
            graphs = []

    # NEW LETTERS
    for i in range(6):
        generate_new_letters(letters_dict, autoencoder, letters)


if __name__ == "__main__":
    __main__()
