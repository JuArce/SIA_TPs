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

    standardize_data = StandardScaler().fit_transform(data)

    # Principal components
    pca = PCA()
    principal_components = pca.fit_transform(standardize_data)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=components, index=df.index.values)
    eigenvectors = pd.DataFrame(data=np.transpose(pca.components_), columns=eigenvectors_c)
    # Cargas del primer autovector
    f_e = np.transpose(pca.components_)[:, 0]  # primer autovector
    f_e_d = pd.DataFrame(data=np.transpose(pca.components_)[:, 0], index=variables)
    # Pesos de la primera componente principal
    f_c = pd.DataFrame(data=principal_components[:, 0], index=df.index.values, columns=['Component 1'])
    f_c.sort_values(by='Component 1', axis=0, inplace=True)

    # -------------- OJA --------------
    # Con la implementación de oja
    parameters = OjaParameters(config)
    oja = Oja(parameters, len(standardize_data[0]))
    results = oja.train(standardize_data)

    # Primera componente con Oja
    f_c_o = np.matmul(standardize_data, results.w[-1])
    # A cada w obtenido le resto el autovector asociado a la primera componente para comparar el error en cada
    # iteracion de w
    errors = results.w - f_e


if __name__ == '__main__':
    main()
