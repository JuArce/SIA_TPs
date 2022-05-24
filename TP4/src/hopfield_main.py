import itertools
import string
import sys

import numpy as np

from algorithms.Hopfield import Hopfield
from utils.ConfigULK import Config
from utils.HopfieldParameters import HopfieldParameters


def get_all_products(letter_combinations, letters_dict):
    letters_dict_comb = dict()
    for i in range(len(letter_combinations)):
        key = ','.join(letter_combinations[i])
        aux = list(itertools.combinations(letter_combinations[i], 2))
        prod = 0
        for j in range(len(aux)):
            prod += abs(np.dot(letters_dict[aux[j][0]], letters_dict[aux[j][1]]))
        letters_dict_comb[key] = prod

    ordered_dict = dict(sorted(letters_dict_comb.items(), key=lambda item: item[1]))

    return ordered_dict


def main():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 3, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config = Config(f.read())
    f.close()

    # Get letters representation

    letters_array = list(string.ascii_uppercase)
    letters_dict = dict()
    letter_idx = 0

    with open(sys.argv[2], 'r') as inputs_file:
        i = 1
        aux = []
        for line in inputs_file:
            if line != '\n':
                values = line.replace('\n', '').replace(',', '')
                for e in values:
                    aux.append(1 if e == '*' else -1)
                if i % 5 == 0:
                    letters_dict[letters_array[letter_idx]] = np.array(aux)
                    letter_idx += 1
                    aux = []
                i += 1

    ## Get all posibilities of len 4
    letter_combinations = list(itertools.combinations(letters_array, 4))
    letter_products = get_all_products(letter_combinations, letters_dict)

    first = next(iter(letter_products))
    patterns_letters = first.split(',')
    patterns_values = []
    for i in patterns_letters:
        patterns_values.append(letters_dict[i])

    parameters = HopfieldParameters(config)
    hopfield = Hopfield(parameters, np.array(patterns_values))

    hopfield.predict(patterns_values[0])


if __name__ == '__main__':
    main()
