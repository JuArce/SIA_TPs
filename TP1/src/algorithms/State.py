from algorithms.manhattan_distance import manhattan


class State:
    COLS = 3
    ROWS = 3

    def __init__(self, state: str):
        self.id = state
        blank_idx = state.index('0')
        self.col_idx = blank_idx % self.COLS
        self.row_idx = blank_idx // self.ROWS  # The // operator will be available to request floor division unambiguously.


class Heuristic_state(State):

    def __init__(self, state: str, goal: str, heuristic):
        super().__init__(state)
        self.heuristic = heuristic(state, goal)
