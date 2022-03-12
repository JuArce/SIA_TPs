from datetime import datetime
from algorithms.Node import Node

from algorithms.State import State
from algorithms.State import Heuristic_state
from utils.Config import Config
from utils.Plays import Plays
from utils.Results import Results

import sys

x = 1500000
sys.setrecursionlimit(x)


def local_heuristic(config: Config):
    # L lista de nodos que empieza con n0
    # n0 nodo raiz  s es el estado del nodo raiz
    ex, root, frontier, time, result = Plays.initialize_with_heuristic(config)
    deep = 0
    node = local_heuristic_rec(ex, root, frontier, result, config.final_state, deep)

    print("Result: " + result.__str__())
    print(node.state.id)
    time = datetime.now() - time


def local_heuristic_rec(ex, root, frontier, result, goal, deep):
    while len(frontier) > 0:
        # Considerar al nodo n de L cuyo estado tenga
        # el menor valor de heuristica
        frontier.sort(key=lambda n: n.state.heuristic, reverse=True)
        # Remover n de L
        node = frontier.pop()
        successors = Plays.get_moves(node, ex)
        ex.add(node)

        # Si el estado s de n es solucion
        # Fin de busqueda con exito
        if node.state.id == goal:
            deep = node.deep
            result = True
            solution = node
            return solution

        if node.deep > deep:
            deep = node.deep

        # Expandir n de acuerdo a las acciones posibles para
        # el estado que lo etiqueta.
        # Formar una lista de nodos L sucesores
        f_successors = []
        for s in successors:
            state = Heuristic_state(s, goal)
            child = Node(state, node)
            node.children.append(child)
            if child not in ex:
                # L sucesores tiene los nodos
                # obtenidos de la expansion de n.
                f_successors.append(child)

        # LLamar a BusquedaHeuristicaLocal(Lsucesores)
        return local_heuristic_rec(ex, root, f_successors, result, goal, deep)
