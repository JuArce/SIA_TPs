import copy
import random

import math

from utils.selection_parameters import SelectionParameter


def boltzmann(chromosomes: dict, selection_parameter: SelectionParameter):
    elements = copy.deepcopy(chromosomes)
    t = get_temperature(selection_parameter)

    output = dict()

    for e in elements:
        elements[e] = (math.exp(elements[e] / t))

    while len(output.keys()) < selection_parameter.population:
        selected = random.choices(list(elements.items()),
                                  weights=list(elements.values()),
                                  k=(selection_parameter.population - len(output.keys())))
        for k in selected:
            output[k[0]] = chromosomes[k[0]]

            if k[0] in elements:
                elements.pop(k[0])

    return output


def get_temperature(selection_parameter: SelectionParameter):
    tc = selection_parameter.temperature_goal
    to = selection_parameter.initial_temperature
    factor = selection_parameter.decrease_temp_factor * selection_parameter.current_gen
    return tc + (to - tc) * math.exp(- factor)
