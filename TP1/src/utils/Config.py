import json


class Config:
    UNINFORMED_ALGORITHMS = ['bfs', 'dfs', 'vds']
    INFORMED_ALGORITHMS = ['local_heuristic', 'global_heuristic', 'a_star']
    HEURISTICS = ['manhattan', 'adm_heu_2', 'not_adm_heu']

    def __init__(self, string):
        config = json.loads(string)
        self.algorithm = config.get('algorithm')
        self.heuristic = config.get('heuristic') if config.get('heuristic') else None
        self.initial_state = config.get('initial_state')
        self.final_state = config.get('final_state')
        self.max_depth = int(config.get('max_depth')) if config.get('max_depth') else None
        self.max_steps = int(config.get('max_steps')) if config.get('max_steps') else None
        self.qty = int(config.get('qty')) if config.get('qty') else 50

        assert (self.algorithm != ''), 'Algorithm undefined'
        assert ((self.algorithm in self.UNINFORMED_ALGORITHMS and self.heuristic is None)
                or (self.algorithm in self.INFORMED_ALGORITHMS and self.heuristic in self.HEURISTICS)), \
            'Invalid algorithm and heuristic combination'
        # TODO generate initial state and validate well formatted
        # TODO validate final state well formatted
        assert (self.max_depth is None or self.max_depth > 0), 'Max depth must be positive or empty'
        assert (self.max_steps is None or self.max_steps > 0), 'Max steps must be positive or empty'
        # TODO arreglar el texto
        assert (
                self.qty is None or 0 < self.qty <= 500), 'Max quantity of initial moves to must be positive ' \
                                                          'and less than 500 or empty '

    def __str__(self):
        return self.__dict__.__str__()
