from utils.Config_ga import Config
from population.Bag import Bag


class Results:

    def __init__(self, bag: Bag, config: Config):
        self.bag = bag
        self.config = config


