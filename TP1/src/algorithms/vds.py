from datetime import datetime
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
    solution = None

    while vds_depth not in used_depths:
        used_depths.add(vds_depth)
        print("added:" + vds_depth.__str__())
        while len(frontier) > 0:
            node = frontier.pop()
            successors = Plays.get_moves(node, ex)
            ex.add(node)

            # tenemos que pasar al siguiente nodo de la frontera
            if node.state.id == config.final_state:
                deep = node.deep
                result = True
                solution = node
                break

            if node.deep > deep:
                deep = node.deep

            expanded_nodes += 1

            if node.deep > vds_depth:
                for s in successors:
                    state = State(s)
                    child = Node(state, node)
                    node.children.append(child)
                    if child not in ex:
                        frontier.appendleft(child)
                continue

            for s in successors:
                state = State(s)
                child = Node(state, node)
                node.children.append(child)
                if child not in ex:
                    frontier.append(child)
        if result:
            vds_depth = deep - 1
        else:
            vds_depth += 1

    time = datetime.now() - time
    cost = vds_depth

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
