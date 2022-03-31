import random

from utils.selection_parameters import SelectionParameter


def truncated(chromosomes: dict, selection_parameter: SelectionParameter):
    # Ordenamos los mejores primeros
    ordered_dict = dict(sorted(chromosomes.items(), key=lambda item: item[1], reverse=True))
    n = len(ordered_dict) - selection_parameter.k_truncated
    # nos quedamos con los mejores habiendo quitado los k peores y sacamos al azar los que quedan
    return dict(
        random.sample({k: ordered_dict[k] for k in list(ordered_dict)[:n]}.items(), selection_parameter.population))
