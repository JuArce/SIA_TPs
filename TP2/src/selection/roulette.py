import copy
import random

from utils.selection_parameters import SelectionParameter


def roulette(chromosomes: dict, selection_parameter: SelectionParameter):
    elements = copy.deepcopy(chromosomes)
    output = dict()
    while len(output.keys()) < selection_parameter.population:
        selected = random.choices(list(elements.items()),
                                  weights=list(elements.values()),
                                  k=(selection_parameter.population - len(output.keys())))
        output.update(selected)
        for k in selected:
            if k[0] in elements:
                elements.pop(k[0])

    return output
