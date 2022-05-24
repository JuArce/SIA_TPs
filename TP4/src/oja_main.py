import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from algorithms.Oja import Oja
from utils.Oja.ConfigULO import Config
from utils.Oja.OjaParameters import OjaParameters

variables = ['Area', 'GDP', 'Inflation', 'L.expect', 'Military', 'P.growth', 'Unemployment']
components = ['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6', 'Component 7']
eigenvectors_c = ['1', '2', '3', '4', '5', '6', '7']


def main():
    np.set_printoptions(suppress=True)
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 3, 'Missing arguments'
    f = open(sys.argv[1])
    config: Config = Config(f.read())
    f.close()

    df = pd.read_csv(sys.argv[2])
    df.set_index('Country', drop=True, inplace=True)
    data = df.values

    # Standardize the data
    # standardize_data = copy.deepcopy(data)
    #
    # for i in range(len(data[0])):
    #     aux = standardize_data[:, i]
    #     mean_aux = mean(aux)
    #     stdev_aux = stdev(aux)
    #     standardize_data[:, i] = (standardize_data[:, i] - mean_aux) / stdev_aux

    # Calculo con librería

    standardize_data = StandardScaler().fit_transform(data)

    # Principal components
    pca = PCA()
    principal_components = pca.fit_transform(standardize_data)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=components, index=df.index.values)
    eigenvectors = pd.DataFrame(data=np.transpose(pca.components_), columns=eigenvectors_c)
    # Cargas del primer autovector
    f_e = pd.DataFrame(data=np.transpose(pca.components_)[:, 0], index=variables)
    # Pesos de la primera componente principal
    f_c = pd.DataFrame(data=principal_components[:, 0], index=df.index.values, columns=['Component 1'])
    f_c.sort_values(by='Component 1', axis=0, inplace=True)

    # Con la implementación de oja
    parameters = OjaParameters(config)
    oja = Oja(parameters, len(standardize_data[0]))
    vector = oja.train(standardize_data)


if __name__ == '__main__':
    main()
