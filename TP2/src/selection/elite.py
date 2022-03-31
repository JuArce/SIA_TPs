from utils.selection_parameters import SelectionParameter


def elite(chromosomes: dict, selection_parameter: SelectionParameter):
    ordered_dict = dict(sorted(chromosomes.items(), key=lambda item: item[1], reverse=True))
    new_gen: dict = {k: ordered_dict[k] for k in list(ordered_dict)[:selection_parameter.population]}
    return new_gen
