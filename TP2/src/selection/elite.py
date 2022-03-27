from TP2.src.utils.Config import Config


def elite(chromosomes: dict, config: Config):
    ordered_dict = dict(sorted(chromosomes.items(), key=lambda item: item[1], reverse=True))
    new_gen: dict = {k: ordered_dict[k] for k in list(ordered_dict)[:config.population]}
    return new_gen
