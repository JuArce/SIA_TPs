from typing import Optional
from algorithms.State import State


class Node:

    def __init__(self, state: State, parent: Optional['Node']):
        self.state = state
        self.deep = (parent.deep + 1 if parent else 0)
        self.is_visited = False
        self.parent = parent
        self.children = []

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Node) and self.state.id == o.state.id

    def __hash__(self):
        return hash(self.state.id)

    def get_children(self):
        return self.children

    def set_children(self, children):
        self.children = children


class HeuristicNode(Node):

    def __init__(self, state: State, parent: Optional['HeuristicNode']):
        super().__init__(state, parent)
