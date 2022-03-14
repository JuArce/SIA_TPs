from datetime import datetime
from tree.Node import Node
from tree.State import Heuristic_state
from utils.Config import Config
from utils.Plays import Plays
from utils.Results import Results


def local_heuristic(config: Config):
    # L lista de nodos que empieza con n0
    # n0 nodo raiz  s es el estado del nodo raiz
    ex, root, frontier, time, result, heuristic = Plays.initialize_with_heuristic(config)
    expanded_nodes = 0
    deep = 0
    solution = None

    while len(frontier) > 0:
        # Considerar al nodo n de L cuyo estado tenga
        # el menor valor de heuristica
        # frontier.sort(key=lambda n: n.state.heuristic, reverse=True)
        # Remover n de L
        node = frontier.pop()
        successors = Plays.get_moves(node, ex)
        ex.add(node)
        # Si el estado s de n es solucion
        # Fin de busqueda con exito
        if node.state.id == config.final_state:
            deep = node.deep
            result = True
            solution = node
            break

        if node.deep > deep:
            deep = node.deep

        # Expandir n de acuerdo a las acciones posibles para
        # el estado que lo etiqueta.
        # Formar una lista de nodos L sucesores
        expanded_nodes += 1

        f_successors = []
        for s in successors:
            state = Heuristic_state(s, config.final_state, heuristic)
            child = Node(state, node)
            node.children.append(child)
            if child not in ex:
                # L sucesores tiene los nodos
                # obtenidos de la expansion de n.
                f_successors.append(child)
        f_successors.sort(key=lambda n: n.state.heuristic, reverse=True)
        for n in f_successors:
            frontier.append(n)
        # LLamar a BusquedaHeuristicaLocal(Lsucesores)
        # return local_heuristic_rec(ex, root, f_successors, result, goal, deep, heuristic, expanded_nodes + 1)


    time = datetime.now() - time
    cost = deep

    plays_to_win = Plays.get_plays_to_win(solution) if result else None

    results = {
        'config': config,
        "result": result,
        "deep": solution.deep,
        "cost": cost,
        "expandedNodes": expanded_nodes,
        "frontierNodes": len(frontier),
        "time": time,
        "plays_to_win": plays_to_win
    }

    return Results(results)