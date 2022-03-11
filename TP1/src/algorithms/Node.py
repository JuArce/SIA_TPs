class Node:

    def __init__(self, state):
        self.state = state  # que id es (es el estado)
        self.col_idx = state.col_idx
        self.row_idx = state.row_idx
        self.is_visited = False
        self.children = []

    def get_children(self):
        return self.children

    def set_children(self, children):
        self.children = children
