import datetime

from population.Bag import Bag
from utils.Config_ga import Config


class Results:

    def __init__(self, bag: Bag, config: Config, time: datetime):
        self.bag = bag
        self.config = config
        self.time = datetime.datetime.now() - time
