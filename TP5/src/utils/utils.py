from copy import deepcopy

import numpy as np


def to_bin_array(encoded_caracter):
    bin_array = np.zeros((7, 5), dtype=float)
    for row in range(0, 7):
        current_row = encoded_caracter[row]
        for col in range(0, 5):
            bin_array[row][4 - col] = current_row & 1
            current_row >>= 1
    return bin_array  # bin_array = [[...],[...],...]


def mutate_pattern(patterns, bytes_to_change):
    patterns = deepcopy(patterns)
    for p in patterns:
        indexs = np.random.choice(len(p), bytes_to_change, replace=False)
        r = np.random.uniform(low=-0.5, high=0.5, size=len(indexs))
        p[indexs] += r
    return patterns


def resize_letter(x):
    return np.array(np.split(x, 7))


def midpoint(p1_x, p1_y, p2_x, p2_y):
    return np.array([(p1_x + p2_x) / 2, (p1_y + p2_y) / 2])
