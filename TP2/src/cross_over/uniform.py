import random
from utils.Config import Config


def uniform(chromosomes: [str], config: Config):
    aux = [[], []]

    for i in range(len(chromosomes[0])):
        p = random.random()
        # Hay que intercambiar si p < 0.5
        if p < 0.5:
            aux[0].append(chromosomes[1][i])
            aux[1].append(chromosomes[0][i])
        else:
            aux[0].append(chromosomes[0][i])
            aux[1].append(chromosomes[1][i])
    output = ["".join(aux[0]), "".join(aux[1])]

    return output
