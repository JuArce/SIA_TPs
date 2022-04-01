import random

from TP2.src.utils.selection_parameters import SelectionParameter


def rank(chromosomes: dict, selection_parameter: SelectionParameter):
    elements = dict(sorted(chromosomes.items(), key=lambda item: item[1]))
    output = dict()
    while len(output.keys()) < selection_parameter.population:
        weights = list(range(len(elements)))
        selected = random.choices(list(elements.items()),
                                  weights=weights,
                                  k=(selection_parameter.population - len(output.keys())))
        output.update(selected)
        for k in selected:
            if k[0] in elements:
                elements.pop(k[0])

    return output
