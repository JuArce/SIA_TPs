import sys

import numpy as np

from algorithms.VAE import VAE, flatten_set
from src.utils.Config_A_VAE import Config_A_VAE
from src.utils.fonts import font_2
from src.utils.utils import to_bin_array


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 4, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config_A_VAE = Config_A_VAE(f.read())
    f.close()
    #
    # data_directory = sys.argv[2]
    # output_directory = sys.argv[3]
    #
    # filenames = [os.path.join(data_directory, file_i)
    #              for file_i in os.listdir(data_directory)
    #              if '.png' in file_i]
    #
    # imgs = [plt.imread(f) for f in filenames]
    #
    # flattened_set = flatten_set(imgs).astype('float32')
    #
    # original_dim = len(flattened_set[0])
    # latent_dim = config.latent_code_len
    # batch_size = 100

    data = []
    letters_patterns = []
    letters = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']
    for letter in font_2:
        aux = to_bin_array(letter)
        data.append(np.concatenate(aux))
        letters_patterns.append(aux)

    flattened_set = flatten_set(data).astype('float32')

    letters_dict = dict(zip(letters, letters_patterns))
    vae = VAE(config.latent_code_len, len(flattened_set[0]), config.neurons_per_layer)
    vae.train(flattened_set, config.epochs, len(flattened_set))
    train_encoded = vae.encoder.predict(flattened_set, batch_size=len(flattened_set[0]))

    # vae = VAE(config.latent_code_len, len(flattened_set[0]), config.neurons_per_layer)
    # vae.train(flattened_set, config.epochs, len(flattened_set))
    # train_encoded = vae.encoder.predict(flattened_set, batch_size=len(flattened_set[0]))
    print(train_encoded)

    # print(train_encoded)
    # # print


if __name__ == "__main__":
    __main__()
