# Recibe estado inicial, final, límites

# Se fija donde está el 0. Analiza los posibles estados hijos. Crea los estados hijos. Los empieza a recorrar
import numpy as np

import sys

sys.path.append("..")

from datetime import datetime
from utils.Results import Results
from algorithms.Node import Node
from algorithms.State import State
from utils.Plays import Plays
from utils.Config import Config


def bpp(config: Config):
    ex = set()  # Conjunto Ex de nodos explorados. Hashset para guardar los estados ya visitados y accederlos rápidamente

    state = State(config.initial_state)
    root = Node(state, None)  # creación del arbol A

    frontier = [root]  # conjunto F de nodos frontera.
    config = config
    time = datetime.now()  # get initial time

    result = False

    while len(frontier) > 0:
        # extraemos primer nodo n de F
        node = frontier.pop()
        successors = Plays.get_moves(node, ex)
        print(node.state.id)
        # agregamos n si no esta en Ex
        ex.add(node)

        if node.state.id == config.final_state:
            result = True
            break

        for s in successors:
            state = State(s)
            child = Node(state, node)
            node.children.append(child)
            if child not in ex:
                frontier.append(child)

        ## agregar en A todos los nodos y en F solo si no está en ex
        ## reordenar F según el método de búsqueda --> el set se encarga

    time = datetime.now() - time

    results = {
        "config": config,
        "result": result,
        "deep": "deep",
        "cost": "cost",
        "expandedNodes": "plays",
        "frontierNodes": frontier,
        "time": time
    }
    return Results(results)
