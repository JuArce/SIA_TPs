from datetime import datetime
from tree.Node import Node
from tree.State import State
from utils.Config_sm import Config
from utils.Plays import Plays
from utils.Results_sm import Results


def bfs(config: Config):
    ex, root, frontier, time, result = Plays.initialize(config)
    frontier.append(root)
    expanded_nodes = 0
    deep = 0
    solution = None

    while len(frontier) > 0:
        node = frontier.pop()
        successor = Plays.get_moves(node, ex)
        ex.add(node)

        if node.state.id == config.final_state:
            deep = node.deep
            result = True
            solution = node
            break

        if node.deep > deep:
            deep = node.deep

        expanded_nodes += 1

        for s in successor:
            state = State(s)
            child = Node(state, node)
            node.children.append(child)
            if child not in ex:
                frontier.appendleft(child)

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
