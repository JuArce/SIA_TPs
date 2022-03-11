# Recibe estado inicial, final, límites

# Se fija donde está el 0. Analiza los posibles estados hijos. Crea los estados hijos. Los empieza a recorrar
import numpy as np
from ..utils.Results import Results

from datetime import datetime
from Node import Node
from ..utils.Plays import Plays


def __init__(self, config):
    self.plays = []  # jugadas ya realizadas para imprimir mas tarde
    self.ex = set()  # Conjunto Ex de nodos explorados. Hashset para guardar los estados ya visitados y accederlos rápidamente
    self.root = Node(config.initial_state)  # creación del arbol A
    self.frontier = [self.root.state]  # conjunto F de nodos frontera.
    self.deep = 0
    self.cost = 0
    self.config = config
    self.time = datetime.now()  # get initial time

    self.solve()  # resolver el juego


def solve(self):
    result = False
    while self.frontier.size > 0:
        ## extraemos primer nodo n de F
        node = self.frontier.pop()
        ## agregamos n si no esta en Ex
        self.ex.add(node.state)
        if node.state == self.config.final_state:
            result = True
            break
        sucessors = Plays.get_moves(node)

        ## agregar en A todos los nodos y en F solo si no está en ex
        ## reordenar F según el método de búsqueda --> el set se encarga

    self.time = datetime.now() - self.time

    results = {
        "plays": self.plays,
        "config": self.config,
        "result": result,
        "deep": self.deep,
        "cost": self.cost,
        "expandedNodes": self.plays,
        "frontierNodes": self.frontier,
        "time": self.time
    }

    return Results(results)
