import sys

import numpy as np

from algorithms.Hopfield import Hopfield
from utils.ConfigULK import Config
from utils.HopfieldParameters import HopfieldParameters


def main():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 3, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config = Config(f.read())
    f.close()

    x = []
    with open(sys.argv[2], 'r') as inputs_file:
        i = 1
        aux = []
        for line in inputs_file:
            if line != '\n':
                values = line.replace('\n', '').replace(',', '')
                for e in values:
                    aux.append(1 if e == '*' else -1)
                if i % 5 == 0:
                    x.append(aux)
                    aux = []
                i += 1

    x = np.array(x)

    parameters = HopfieldParameters(config)

    hopfield = Hopfield(parameters, x)


if __name__ == '__main__':
    main()
