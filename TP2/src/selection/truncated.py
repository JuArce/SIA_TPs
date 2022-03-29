import random

from TP2.src.utils.Config import Config


def truncated(chromosomes: dict, config: Config):
    # Ordenamos los mejores primeros
    ordered_dict = dict(sorted(chromosomes.items(), key=lambda item: item[1], reverse=True))
    n = len(ordered_dict) - config.k_truncated
    # nos quedamos con los mejores habiendo quitado los k peores y sacamos al azar los que quedan
    return dict(random.sample({k: ordered_dict[k] for k in list(ordered_dict)[:n]}.items(), config.population))
