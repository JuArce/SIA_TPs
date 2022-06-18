from copy import deepcopy

import numpy as np


def to_bin_array(encoded_caracter):
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(0, 7):
        current_row = encoded_caracter[row]
        for col in range(0, 5):
            bin_array[row][4 - col] = current_row & 1
            current_row >>= 1
    return bin_array  # bin_array = [[...],[...],...]


def mutate_pattern(pattern, bytes_to_change):
    indexs = np.random.choice(len(pattern), bytes_to_change, replace=False)
    p = deepcopy(pattern)
    for i in indexs:
        p[i] = 1 if p[i] == 0 else 0

    return p


def resize_letter(x):
    return np.array(np.split(x, 7))
