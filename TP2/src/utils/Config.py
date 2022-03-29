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
        self.multiple_cross_points = config.get('multiple_cross_points')  # TODO : agregar validaciones

        self.population = int(config.get('population'))

        self.limit_time = int(config.get('limit_time'))  # TODO : agregar validaciones
        self.generations_quantity = int(config.get('generations_quantity'))  # TODO: agregar validaciones
        self.mutation_probability = float(config.get('mutation_probability'))  # TODO: agregar validaciones
        self.k_truncated = int(config.get('k_truncated'))  # TODO: agregar validaciones.

        if self.population == '':
            self.population = 500

        assert (self.selection_algorithm != ''), 'Selection algorithm undefined'
        assert (self.selection_algorithm in self.SELECTION_ALGORITHMS), 'Invalid selection algorithm'

        assert (self.cross_over_algorithm != ''), 'Cross over algorithm undefined'
        assert (self.cross_over_algorithm in self.CROSS_OVER_ALGORITHMS), 'Invalid cross over algorithm'

    def __str__(self):
        return self.__dict__.__str__()
