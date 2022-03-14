import collections
from datetime import datetime
from typing import Optional
from tree.Node import Node
from tree.State import State
from utils.Config import Config
from utils.Plays import Plays
from utils.Results import Results


def vds(config: Config):
    ex, root, frontier, time, result = Plays.initialize(config)
    expanded_nodes = 0
    deep = 0
    vds_depth = int(config.initial_depth)
    ex = dict()
    solution: Optional[Node] = None
    frontier_aux = collections.deque()
    frontier_aux.appendleft(root)

    while len(frontier_aux) > 0 and vds_depth > 0:
        while len(frontier_aux) > 0:
            frontier.appendleft(frontier_aux.pop())

        while len(frontier) > 0:
            node = frontier.pop()

            successors = Plays.get_moves_from_dict(node, ex)

            d = ex.get(node.state.id)
            if d is None or node.deep < d:
                ex[node.state.id] = node.deep
                expanded_nodes += 1

            # La profundidad de la solucion tiene que ser siempre menor que la que ya se habia obtenido
            if not solution or (solution and node.deep < solution.deep):
                if node.state.id == config.final_state:
                    solution = node
                    deep = node.deep
                    result = True
                    break

            for s in successors:
                state = State(s)
                child = Node(state, node)
                node.children.append(child)
                if child.deep > vds_depth:
                    frontier_aux.appendleft(child)
                else:
                    frontier.append(child)

        if not solution:  # no encontré solución y tengo que agrandar las profundidades
            vds_depth += 1

        elif result:  # encontre otra solución distinta a la que tenía. Sigo bajando la profundidad y sigo probando
            result = False
            vds_depth = deep - 1

    time = datetime.now() - time
    cost = deep

    plays_to_win = []
    if solution:
        result = True
        plays_to_win = Plays.get_plays_to_win(solution)

    results = {
        'config': config,
        "result": result,
        "deep": deep,
        "cost": cost,
        "expandedNodes": expanded_nodes,
        "frontierNodes": len(frontier),
        "time": time,
        "plays_to_win": plays_to_win
    }
    return Results(results)
