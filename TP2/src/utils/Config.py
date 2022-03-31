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

        if self.selection_algorithm == 'tournament':
            assert (config.get('tournament_probability') != ''), 'Missing \'u\' probability'
            assert (float(config.get('tournament_probability')) >= 0.5 and float(config.get(
                'tournament_probability')) <= 1), '\'u\' must be between 0.5 and 1'
            self.tournament_probability = float(config.get('tournament_probability'))
        else:
            self.tournament_probability = None

        if self.population == '':
            self.population = 500

        assert (config.get('max_unchanged_generations') != ''), 'Must define max unchanged generations'
        assert (int(config.get('max_unchanged_generations')) > 0 and int(
            config.get(
                'max_unchanged_generations')) <= 50000), 'The value of the max unchanged generations ' \
                                                         'must be between 0 and 500 '
        self.max_unchanged_generations = int(config.get('max_unchanged_generations'))

        assert (config.get('unchanged_percentage') != ''), 'Must define unchanged percentage of generations'
        assert (float(config.get('unchanged_percentage')) > 0 and float(
            config.get(
                'unchanged_percentage')) < 1), 'The value of the unchanged percentage must ' \
                                               'be greater than 0 and less than 1 '
        self.unchanged_percentage = float(config.get('unchanged_percentage'))

        assert (config.get('max_unchanged_fitness_generations') != ''), 'Must define max unchanged fitness generations'
        assert (int(config.get('max_unchanged_fitness_generations')) > 0 and int(
            config.get(
                'max_unchanged_fitness_generations')) <= 500), 'The value of the max unchanged generations ' \
                                                               'fitness in generations must be between 0 and 500 '
        self.max_unchanged_fitness_generations = int(config.get('max_unchanged_fitness_generations'))

        assert (self.selection_algorithm != ''), 'Selection algorithm undefined'
        assert (self.selection_algorithm in self.SELECTION_ALGORITHMS), 'Invalid selection algorithm'

        assert (self.cross_over_algorithm != ''), 'Cross over algorithm undefined'
        assert (self.cross_over_algorithm in self.CROSS_OVER_ALGORITHMS), 'Invalid cross over algorithm'

        self.temperature = config.get('temperature')
        self.temperature_goal = config.get('temperature_goal')
        self.decrease_temp_factor = config.get('decrease_temp_factor')

    def __str__(self):
        return self.__dict__.__str__()
