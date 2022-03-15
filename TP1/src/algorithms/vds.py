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

    step = int(config.initial_depth)

    pending_frontier = collections.deque()
    pending_frontier.append(root)
    expanded_nodes = 0

    ex = dict()
    ex[root.state.id] = 0
    result = False

    solution: Optional[Node] = None

    while len(pending_frontier) > 0:

        n = pending_frontier.popleft()

        max_depth = n.deep + step

        if solution is not None and n.deep >= solution.deep:
            pending_frontier.appendleft(n)
            break

        frontier.append(n)

        while frontier:
            node: Node = frontier.pop()

            if node.deep >= max_depth:
                pending_frontier.append(node)
                continue

            if node.state.id == config.final_state:
                result = True
                if solution is None or solution.deep > node.deep:
                    solution = node
                max_depth = solution.deep
                continue

            expanded_nodes += 1
            for s in Plays.get_moves_from_dict(node, ex):
                state = State(s)
                child = Node(state, node)
                node.children.append(child)

                if child.state.id not in ex.keys() or child.deep < ex[child.state.id]:
                    ex[child.state.id] = child.deep
                    frontier.append(child)

    plays_to_win = Plays.get_plays_to_win(solution) if result else None

    time = datetime.now() - time
    cost = solution.deep
    results = {
        'config': config,
        "result": result,
        "deep": solution.deep,
        "cost": cost,
        "expandedNodes": expanded_nodes,
        "frontierNodes": len(frontier) + len(pending_frontier),
        "time": time,
        "plays_to_win": plays_to_win
    }
    return Results(results)
