import json


class Config_A_VAE:

    def __init__(self, string):
        config = json.loads(string)
        self.latent_code_len = int(config.get('latent_code_len'))
        self.epochs = int(config.get('epochs'))
        self.neurons_per_layer = config.get('neurons_per_layer')
