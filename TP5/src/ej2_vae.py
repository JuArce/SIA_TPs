import sys

from algorithms.VAE import VAE, flatten_set
from utils.Config_A import Config_A


def __main__():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 2, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config_A = Config_A(f.read())
    f.close()
    data = []

    flattened_set = flatten_set(properties.training_set).astype('float32')
    latent_index = int((len(properties.neurons_per_layer)) / 2)
    vae = VAE(properties.neurons_per_layer[latent_index], len(properties.training_set[0]),
              properties.neurons_per_layer[:latent_index])
    vae.train(flattened_set, properties.epochs, len(properties.training_set))
    # train_encoded = vae.encoder.predict(flattened_set, batch_size=len(properties.training_set))[0]
    # print(train_encoded)
    # print


if __name__ == "__main__":
    __main__()
