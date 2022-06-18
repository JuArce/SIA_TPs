import json


class Config_A:
    PERCEPTRON_ALGORITHMS = ["multi_layer_perceptron"]
    FUNCTIONS = ['logistic', 'tanh']

    def __init__(self, string):
        config = json.loads(string)
        self.algorithm = config.get('algorithm')
        self.max_iter = int(config.get('max_iter'))
        self.latent_code_len = int(config.get('latent_code_len'))
        self.learning_rate = float(config.get('learning_rate'))
        self.betha = float(config.get('betha'))
        self.function = config.get('function')
        self.layers = config.get('layers')
        self.min_error = float(config.get('min_error'))
        self.k = int(config.get('k'))

    def __str__(self):
        label = [self.algorithm]
        return ''.join(label)
