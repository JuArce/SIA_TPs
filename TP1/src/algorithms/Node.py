from typing import Optional
from algorithms.State import State
from algorithms.State import Heuristic_state


class Node:

    def __init__(self, state: State or Heuristic_state, parent: Optional['Node']):
        self.state = state
        # profundidad de la solución
        self.deep = (parent.deep + 1 if parent else 0)
        self.is_visited = False
        self.parent = parent
        self.children = []

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Node) and self.state.id == o.state.id

    def __hash__(self):
        return hash(self.state.id)
