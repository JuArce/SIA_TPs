import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from algorithms.Kohonen import Kohonen
from utils.Kohonen.ConfigULK import Config
from utils.Kohonen.KohonenParameters import KohonenParameters


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
    standardize_data = StandardScaler().fit_transform(data)

    parameters = KohonenParameters(config)
    kohonen = Kohonen(parameters, standardize_data)
    kohonen.train(standardize_data)
    kohonen_results = kohonen.get_results(standardize_data, df.index)


if __name__ == '__main__':
    main()
