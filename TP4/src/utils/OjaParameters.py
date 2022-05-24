from utils.ConfigULO import Config


class OjaParameters:

    def __init__(self, config: Config):
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate
