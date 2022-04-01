import random
from TP2.src.utils.Config import Config


def multiple(chromosomes: [str], config: Config):
    points = config.multiple_cross_points
    p = random.sample(range(len(chromosomes[0]) - 1), points)
    p.sort()

    output: [str] = [[], []]
    prev = 0
    idx_0 = 0
    idx_1 = 1
    for i in p:
        output[0].append(chromosomes[idx_0][prev: i])
        output[1].append(chromosomes[idx_1][prev: i])
        prev = i
        idx_0, idx_1 = idx_1, idx_0
    output[0].append(chromosomes[idx_0][prev: len(chromosomes[0])])
    output[1].append(chromosomes[idx_1][prev: len(chromosomes[1])])

    output[0] = ''.join(output[0])
    output[1] = ''.join(output[1])
    return output
