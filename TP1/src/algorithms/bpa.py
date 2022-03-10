# Recibe estado inicial, final, límites

# Se fija donde está el 0. Analiza los posibles estados hijos. Crea los estados hijos. Los empieza a recorrar

def __init__(self, config):
    buildInitialMatrix(config.initial_state)
    self.plays = []  ##jugadas ya realizadas

    self.solve()
