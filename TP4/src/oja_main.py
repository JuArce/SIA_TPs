import copy
import sys
from statistics import mean, stdev

import numpy as np
import pandas as pd

from algorithms.Oja import Oja
from utils.Oja.ConfigULO import Config
from utils.Oja.OjaParameters import OjaParameters


def main():
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 3, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config = Config(f.read())
    f.close()

    np.set_printoptions(suppress=True)

    df = pd.read_csv(sys.argv[2])
    df.set_index('Country', drop=True, inplace=True)
    data = df.values

    # Standardize the data
    standardize_data = copy.deepcopy(data)

    for i in range(len(data[0])):
        aux = standardize_data[:, i]
        mean_aux = mean(aux)
        stdev_aux = stdev(aux)
        standardize_data[:, i] = (standardize_data[:, i] - mean_aux) / stdev_aux

    parameters = OjaParameters(config)
    oja = Oja(parameters, len(standardize_data[0]))
    vector = oja.train(standardize_data)





if __name__ == '__main__':
    main()
