from datetime import datetime
from typing import Optional
from utils.Config import Config
from algorithms.Node import Node
from algorithms.State import State
from utils.Plays import Plays
from utils.Results import Results


def vds(config: Config):
    ex, root, frontier, time, result = Plays.initialize(config)
    expanded_nodes = 0
    deep = 0
    used_depths = set()
    vds_depth = int(config.initial_depth)
    solution: Optional[Node] = None

    while vds_depth not in used_depths:
        used_depths.add(vds_depth)
        while len(frontier) > 0:
            node = frontier.pop()

            successors = []
            # La profundidad de la solucion tiene que ser siempre menor que la que ya se habia obtenido
            if not solution or (solution and node.deep < solution.deep):
                if node.state.id == config.final_state:
                    solution = node
                    deep = node.deep
                    result = True
                    break

            successors = Plays.get_moves(node, ex)
            ex.add(node)
            expanded_nodes += 1

            for s in successors:
                state = State(s)
                child = Node(state, node)
                node.children.append(child)
                if child not in ex:
                    if child.deep <= vds_depth:
                        frontier.appendleft(child)
                    else:
                        frontier.append(child)

        if result:
            print("Bajando deep" + deep.__str__())
            vds_depth = deep - 1
        else:
            print("Subo deep " + vds_depth.__str__())
            vds_depth += 1

    time = datetime.now() - time
    cost = deep

    plays_to_win = Plays.get_plays_to_win(solution) if result else None

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
