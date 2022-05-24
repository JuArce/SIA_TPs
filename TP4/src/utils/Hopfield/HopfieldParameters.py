from utils.Kohonen.ConfigULK import Config


class HopfieldParameters:

    def __init__(self, config: Config):
        self.max_iterations = config.max_iterations
