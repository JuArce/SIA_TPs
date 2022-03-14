import json


class Config:
    UNINFORMED_ALGORITHMS = ['bfs', 'dfs', 'vds']
    INFORMED_ALGORITHMS = ['local_heuristic', 'global_heuristic', 'a_star']
    HEURISTICS = ['manhattan', 'hamming', 'not_adm_heu']

    def __init__(self, string):
        config = json.loads(string)
        self.algorithm = config.get('algorithm')
        self.heuristic = config.get('heuristic') if config.get('heuristic') else None
        self.initial_state = config.get('initial_state') if config.get('initial_state') else None
        self.final_state = config.get('final_state')
        self.initial_depth = config.get('initial_depth')
        self.qty = int(config.get('qty')) if config.get('qty') else None

        assert (self.algorithm != ''), 'Algorithm undefined'
        assert ((self.algorithm in self.UNINFORMED_ALGORITHMS and self.heuristic is None)
                or (self.algorithm in self.INFORMED_ALGORITHMS and self.heuristic in self.HEURISTICS)), \
            'Invalid algorithm and heuristic combination'
        # TODO generate initial state and validate well formatted
        # TODO validate final state well formatted
        # TODO arreglar el texto
        assert (
                (self.initial_state is not None and self.qty is None) or
                (
                        self.initial_state is None and self.qty is not None)), 'You must specify initial_state ' \
                                                                               'or quantity of moves to ' \
                                                                               'randomize the board '

        assert (
            (self.initial_state is not None or self.qty is not None)), 'Initial state or quantity of moves' \
                                                                       'must not be None'

        assert (
                self.qty is None or 0 < self.qty <= 100000), 'Max quantity of initial moves to must be positive ' \
                                                             'and less than 100000 or empty '

    def __str__(self):
        return self.__dict__.__str__()
