import copy
import random

from TP2.src.utils.Config import Config


def roulette(chromosomes: dict, config: Config):
    elements = copy.deepcopy(chromosomes)
    output = dict()
    while len(output.keys()) < config.population:
        selected = random.choices(list(elements.items()),
                                  weights=list(elements.values()),
                                  k=(config.population - len(output.keys())))
        output.update(selected)
        for k in selected:
            if k[0] in elements:
                elements.pop(k[0])

    return output
