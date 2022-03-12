from TP1.src.algorithms.Node import Node
import numpy as np


class Plays:
    @classmethod
    def get_moves(cls, node: Node, ex: set):
        matrix = [(node.state.id[0:3]),
                  (node.state.id[3:6]),
                  (node.state.id[6:9])]

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


Plays.get_moves = staticmethod(Plays.get_moves)
