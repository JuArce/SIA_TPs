import sys

import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt

from algorithms.VAE import VAE
from utils.Config_A_VAE import Config_A_VAE


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 2, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config_A_VAE = Config_A_VAE(f.read())
    f.close()

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    data_size = 28

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    vae = VAE(config.latent_code_len, len(x_train[0]), config.neurons_per_layer)
    vae.train(x_train, config.epochs, len(x_train))
    train_encoded = vae.encoder.predict(x_test, batch_size=len(x_test[0]))
    train_encoded = train_encoded[0]

    plt.figure(figsize=(6, 6))
    plt.scatter(train_encoded[:, 0], train_encoded[:, 1], c=y_test, cmap='viridis')
    plt.colorbar()
    plt.show()

    # Display a 2D manifold of the digits
    n = 15
    digit_size_y = data_size
    digit_size_x = data_size
    figure = np.zeros((digit_size_y * n, digit_size_x * n))
    # We will sample n points within [-15, 15] standard deviations
    std = 2
    grid_x = np.linspace(-std, std, n)
    grid_y = np.linspace(-std, std, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size_y, digit_size_x)
            figure[i * digit_size_y: (i + 1) * digit_size_y,
            j * digit_size_x: (j + 1) * digit_size_x] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


if __name__ == "__main__":
    __main__()
