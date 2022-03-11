class Node:

    def __init__(self, state):
        self.state = state  # que id es (es el estado)
        self.col_idx = state.
        self.row_idx = state.row
        self.is_visited = False
        self.children = []

    def get_children(self):
        return self.children

    def set_children(self, children):
        self.children = children
