import collections
from datetime import datetime
from TP1.src.algorithms.Node import Node
from TP1.src.algorithms.State import State


class Plays:

    # TODO: Comentar métodos y agregar al método los tipos de qué devuelven
    @classmethod
    def initialize(cls, initial_state: str):
        ex = set()
        state = State(initial_state)
        root = Node(state, None)
        frontier = [root]
        time = datetime.now()
        result = False
        return ex, root, frontier, time, result

    @classmethod
    def get_moves(cls, node: Node, ex: set):

        if node in ex:
            return []

        moves = []

        if node.state.col_idx < 2:
            moves.append(cls.move_right(node))
            # Muevo solo a la derecha
        if node.state.col_idx > 0:
            moves.append(cls.move_left(node))

        if node.state.row_idx > 0:
            moves.append(cls.move_up(node))

        if node.state.row_idx < 2:
            moves.append(cls.move_down(node))

        return moves

    @classmethod
    def move_right(cls, node: Node):
        state_id = list(node.state.id)
        blank_idx = state_id.index('0')
        state_id[blank_idx], state_id[blank_idx + 1] = state_id[blank_idx + 1], state_id[blank_idx]
        return ''.join(state_id)

    @classmethod
    def move_left(cls, node: Node):
        state_id = list(node.state.id)
        blank_idx = state_id.index('0')
        state_id[blank_idx], state_id[blank_idx - 1] = state_id[blank_idx - 1], state_id[blank_idx]
        return ''.join(state_id)

    @classmethod
    def move_up(cls, node: Node):
        state_id = list(node.state.id)
        blank_idx = state_id.index('0')
        state_id[blank_idx], state_id[blank_idx - 3] = state_id[blank_idx - 3], state_id[blank_idx]
        return ''.join(state_id)

    @classmethod
    def move_down(cls, node: Node):
        state_id = list(node.state.id)
        blank_idx = state_id.index('0')
        state_id[blank_idx], state_id[blank_idx + 3] = state_id[blank_idx + 3], state_id[blank_idx]
        return ''.join(state_id)

    @classmethod
    def get_plays_to_win(cls, node: Node):

        plays = collections.deque()

        while node is not None:
            plays.appendleft(node.state.id)
            node = node.parent

        return plays
