from ..algorithms.Node import Node
import numpy as np


class Plays:
    @classmethod
    def get_moves(cls, node: Node, ex: set):

        matrix = np.array([(node.state[0:3]),
                           (node.state[3:6]),
                           (node.state[6:9])])

        if node.state in set:
            return []

        agent_idx = np.where(matrix == '0')








Plays.getMoves = staticmethod(Plays.getMoves)
