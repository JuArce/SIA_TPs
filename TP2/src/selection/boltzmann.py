import math

from utils.selection_parameters import SelectionParameter


def boltzmann(chromosomes: dict, selection_parameter: SelectionParameter):
    t = get_temperature(selection_parameter)


def get_temperature(selection_parameter: SelectionParameter):
    tc = selection_parameter.temperature_goal
    to = selection_parameter.initial_temperature
    factor = selection_parameter.decrease_temp_factor * selection_parameter.current_gen
    return tc + (to - tc) * math.exp(- factor)
