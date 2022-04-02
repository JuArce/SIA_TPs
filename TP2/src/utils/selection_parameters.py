from utils.Config_ga import Config


class SelectionParameter:

    def __init__(self, config: Config):
        self.current_gen = 0
        self.k_truncated = config.k_truncated
        self.population = config.population
        self.tournament_probability = config.tournament_probability
        self.initial_temperature = config.temperature
        self.temperature_goal = config.temperature_goal
        self.decrease_temp_factor = config.decrease_temp_factor
