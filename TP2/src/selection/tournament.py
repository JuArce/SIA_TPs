import random

from TP2.src.utils.Config import Config


def tournament(chromosomes: dict, config: Config):
    u: float = config.tournament_probability
    chromosomes_list: list = [*chromosomes.keys()]
    new_gen: dict = dict()

    while len(new_gen) < config.population:
        couple_a = random.sample(chromosomes_list, 2)
        couple_b = random.sample(chromosomes_list, 2)
        couples: [[str]] = [couple_a, couple_b]
        winners: [str] = []

        while len(couples) > 0:
            c = couples.pop()
            less_fit, fittest = get_order(chromosomes, c[0], c[1])

            r = random.uniform(0, 1)

            # selecciono el más apto
            if r < u:
                winners.append(fittest)
            else:
                winners.append(less_fit)

            if len(winners) == 2:
                couples.append(winners.copy())
                winners.clear()
        new_gen[winners[0]] = chromosomes[winners[0]]

    return new_gen


def get_order(chromosomes: dict, chromosome_a: str, chromosome_b: str):
    if chromosomes[chromosome_a] < chromosomes[chromosome_b]:
        return chromosome_a, chromosome_b
    else:
        return chromosome_b, chromosome_a
