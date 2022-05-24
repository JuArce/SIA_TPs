from utils.Kohonen.ConfigULK import Config


class KohonenParameters:

    def __init__(self, config: Config):
        self.output_layer_len = config.output_layer_qty
        self.epochs = config.max_iterations
        self.initial_radius = config.initial_radius
        self.learning_rate = config.learning_rate
