# Recibe estado inicial, final, límites
#  Se fija donde está el 0. Analiza los posibles estados hijos. Crea los estados hijos. Los empieza a recorrar
from datetime import datetime
from algorithms.Node import Node
from algorithms.State import State
from utils.Config import Config
from utils.Plays import Plays
from utils.Results import Results


def dfs(config: Config):
    # Conjunto Ex de nodos explorados. Hashset para guardar los estados ya visitados y accederlos rápidamente
    # Conjunto F de nodos frontera.

    ex, root, frontier, time, result = Plays.initialize(config)
    expanded_nodes = 0
    deep = 0
    solution = None

    while len(frontier) > 0:
        # extraemos primer nodo n de F
        node = frontier.pop()
        successors = Plays.get_moves(node, ex)
        # print(node.state.id)
        # agregamos n si no esta en Ex
        ex.add(node)

        if node.state.id == config.final_state:
            deep = node.deep
            result = True
            solution = node
            break

        if node.deep > deep:
            deep = node.deep

        for s in successors:
            state = State(s)
            child = Node(state, node)
            node.children.append(child)
            if child not in ex:
                frontier.append(child)

        # agregar en A todos los nodos y en F solo si no está en ex
        # reordenar F según el método de búsqueda --> el set se encarga

    time = datetime.now() - time
    cost = deep  # en el caso del dfs como es uniforme son iguales

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

#  print('Time: ' + time.__str__())
# print('Res: ' + result.__str__())
