import collections
from datetime import datetime
from algorithms.Node import Node
from algorithms.State import State
from utils.Config import Config
import random


class Plays:
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3

    ROWS = 3
    COLS = 3

    # TODO: Comentar métodos y agregar al método los tipos de qué devuelven
    @classmethod
    def initialize(cls, config: Config):
        ex = set()
        state = State(config.initial_state)
        root = Node(state, None)
        if config.algorithm in ['bfs']:
            frontier = collections.deque()
            frontier.append(root)
        else:
            frontier = [root]
        time = datetime.now()
        result = False
        return ex, root, frontier, time, result

    """
    get_valid_moves(cls,row_idx,col_idx)

    @:param Recibe como parámetro los indices de la matriz de dónde se encuentra el agente 
    (dónde se encuentra el 0) e
    @:returns Devuelve un array de punteros a función con los posibles movimientos a ejecutar
    """

    @classmethod
    def get_moves(cls, node: Node, ex: set):
        if node in ex:
            return []

        moves_to_do = cls.get_valid_moves(node.state.row_idx, node.state.col_idx)
        moves = []

        for m in moves_to_do:
            moves.append(m(node.state.id))
        return moves

    """
    get_valid_moves(cls,row_idx,col_idx)
    
    @:param Recibe como parámetro los indices de la matriz de dónde se encuentra el agente 
    (dónde se encuentra el 0) e
    @:returns Devuelve un array de punteros a función con los posibles movimientos a ejecutar
    """

    @classmethod
    def get_valid_moves(cls, row_idx, col_idx) -> []:
        moves = []

        if col_idx < 2:
            moves.append(valid_moves[cls.MOVE_RIGHT])
            # Muevo solo a la derecha
        if col_idx > 0:
            moves.append(valid_moves[cls.MOVE_LEFT])

        if row_idx > 0:
            moves.append(valid_moves[cls.MOVE_UP])

        if row_idx < 2:
            moves.append(valid_moves[cls.MOVE_DOWN])

        return moves

    @classmethod
    def move_right(cls, state_id: str):
        aux = list(state_id)
        blank_idx = aux.index('0')
        aux[blank_idx], aux[blank_idx + 1] = aux[blank_idx + 1], aux[blank_idx]
        return ''.join(aux)

    @classmethod
    def move_left(cls, state_id: str):
        aux = list(state_id)
        blank_idx = aux.index('0')
        aux[blank_idx], aux[blank_idx - 1] = aux[blank_idx - 1], aux[blank_idx]
        return ''.join(aux)

    @classmethod
    def move_up(cls, state_id: str):
        aux = list(state_id)
        blank_idx = aux.index('0')
        aux[blank_idx], aux[blank_idx - 3] = aux[blank_idx - 3], aux[blank_idx]
        return ''.join(aux)

    @classmethod
    def move_down(cls, state_id: str):
        aux = list(state_id)
        blank_idx = aux.index('0')
        aux[blank_idx], aux[blank_idx + 3] = aux[blank_idx + 3], aux[blank_idx]
        return ''.join(aux)

    @classmethod
    def get_plays_to_win(cls, node: Node):
        plays = collections.deque()

        while node is not None:
            plays.appendleft(node.state.id)
            node = node.parent

        return plays

    @classmethod
    def build_initial_play(cls, qty: int):
        random.seed(datetime.now())

        initial_state, blank_idx = "123456780", 8

        for x in enumerate(range(qty)):
            blank_idx = initial_state.index('0')
            col_idx = blank_idx % cls.COLS
            row_idx = blank_idx // cls.ROWS
            moves_to_do = cls.get_valid_moves(row_idx, col_idx)
            initial_state = moves_to_do[random.randint(0, len(moves_to_do) - 1)](initial_state)

        return initial_state


valid_moves = [Plays.move_left, Plays.move_right, Plays.move_up, Plays.move_down]
