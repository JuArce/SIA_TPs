class Results:

    def __init__(self, results):
        self.plays = results.plays
        self.config = results.config
        self.result = results.result
        self.deep = results.deep
        self.cost = results.cost
        self.expandedNodes = results.expandedNodes  ##jugadas hechas
        self.frontierNodes = results.frontierNodes  ##nodos que no se llegaron a expandir
        self.time = results.time
