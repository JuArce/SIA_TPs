import sys

import numpy as np
from matplotlib import pyplot as plt

from algorithms.VAE import VAE, flatten_set
from utils.Config_A_VAE import Config_A_VAE
from utils.fonts import font_2
from utils.utils import to_bin_array


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 2, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config_A_VAE = Config_A_VAE(f.read())
    f.close()

    data = []
    letters_patterns = []
    for letter in font_2:
        aux = to_bin_array(letter)
        data.append(np.concatenate(aux))
        letters_patterns.append(aux)

    flattened_set = flatten_set(data).astype('float32')

    vae = VAE(config.latent_code_len, len(flattened_set[0]), config.neurons_per_layer)
    vae.train(flattened_set, config.epochs, len(flattened_set))
    train_encoded = vae.encoder.predict(flattened_set, batch_size=len(flattened_set[0]))

    train_encoded = train_encoded[0]
    plt.figure(figsize=(6, 6))
    plt.scatter(train_encoded[:, 0], train_encoded[:, 1], cmap='viridis')
    plt.colorbar()
    plt.show()

    # Display a 2D manifold of the digits
    n = 20
    digit_size_y = 7
    digit_size_x = 5
    figure = np.zeros((digit_size_y * n, digit_size_x * n))
    # We will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-1, 1, n)
    grid_y = np.linspace(-1, 1, n)

    graphs = []
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size_y, digit_size_x)
            graphs.append(digit)
            figure[i * digit_size_y: (i + 1) * digit_size_y,
            j * digit_size_x: (j + 1) * digit_size_x] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys")
    plt.show()
    # SeaGraphV2.graph_multi_heatmap(graphs, c_map="Greys", cols=n)


if __name__ == "__main__":
    __main__()
