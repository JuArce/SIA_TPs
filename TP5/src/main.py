import sys

import numpy as np

from algorithms.Autoencoder import Autoencoder
from utils import SeaGraphV2
from utils.Config_A import Config_A
from utils.fonts import font_2
from utils.utils import to_bin_array, resize_letter, midpoint


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 2, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config_A = Config_A(f.read())
    f.close()
    data = []
    letters_patterns = []
    # letters = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    #            'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']
    letters = ['A', 'B', 'C', 'D', 'E', '_']
    for letter in font_2:
        aux = to_bin_array(letter)
        data.append(np.concatenate(aux))
        letters_patterns.append(aux)

    letters_dict = dict(zip(letters, letters_patterns))

    autoencoder = Autoencoder(config, len(data[0]), config.layers, config.latent_code_len)
    autoencoder.train(data, data)

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
    graphs = []
    letter_1 = 'A'
    letter_2 = 'B'

    letter_1_array = np.concatenate(letters_dict[letter_1])
    letter_2_array = np.concatenate(letters_dict[letter_2])
    letter_1_encode = autoencoder.encode(letter_1_array)
    letter_2_encode = autoencoder.encode(letter_2_array)

    graphs.extend([letters_dict[letter_1], letters_dict[letter_2]])

    letter_3_encode = midpoint(letter_1_encode[0], letter_1_encode[1], letter_2_encode[0], letter_2_encode[1])
    letter_3_decode = autoencoder.decode(letter_3_encode)
    res = np.array(list(map(resize_letter, [letter_3_decode])))
    graphs.append(res[0])

    letter_1_decode = autoencoder.decode(letter_1_encode)
    res = np.array(list(map(resize_letter, [letter_1_decode])))
    graphs.append(res[0])

    letter_2_decode = autoencoder.decode(letter_2_encode)
    res = np.array(list(map(resize_letter, [letter_2_decode])))
    graphs.append(res[0])

    SeaGraphV2.graph_multi_heatmap(graphs, title='New Letter', c_map="Greys", cols=3)


if __name__ == "__main__":
    __main__()
