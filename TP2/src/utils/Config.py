import json


class Config:
    SELECTION_ALGORITHMS = ["boltzmann",
                            "elite",
                            "rank",
                            "roulette",
                            "tournament",
                            "truncated"]

    CROSS_OVER_ALGORITHMS = [
        "multiple",
        "simple",
        "uniform"]

    def __init__(self, string):
        config = json.loads(string)
        self.selection_algorithm = config.get('selection_algorithm')
        self.cross_over_algorithm = config.get('cross_over_algorithm')

        self.population = config.get('population')

        if self.population == '':
            self.population = 500

        assert (self.selection_algorithm != ''), 'Selection algorithm undefined'
        assert (self.selection_algorithm in self.SELECTION_ALGORITHMS), 'Invalid selection algorithm'

        assert (self.cross_over_algorithm != ''), 'Cross over algorithm undefined'
        assert (self.cross_over_algorithm in self.CROSS_OVER_ALGORITHMS), 'Invalid cross over algorithm'

    def __str__(self):
        return self.__dict__.__str__()
