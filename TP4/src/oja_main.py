import sys

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from algorithms.Oja import Oja
from utils import SeaGraph
from utils.Oja.ConfigULO import Config
from utils.Oja.OjaParameters import OjaParameters

variables = ['Area', 'GDP', 'Inflation', 'L.expect', 'Military', 'P.growth', 'Unemployment']
components = ['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6', 'Component 7']
eigenvectors_c = ['1', '2', '3', '4', '5', '6', '7']
sns.set_color_codes("pastel")


def main():
    np.set_printoptions(suppress=True)
    print('Argument List:', str(sys.argv))
    assert len(sys.argv) == 4, 'Missing arguments'
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
    errors = abs(results.w[-1]) - abs(f_e)
    f = open(sys.argv[3], 'w')
    aux = [f_e, results.w[-1], abs(errors)]
    f.write('Eigenvector\n')
    aux = pd.DataFrame(data=np.transpose(aux), index=df.columns.values, columns=['Library', 'Oja', 'Error'])
    f.write(aux.to_string())
    f.write('\n\n#########################\n\n')

    errors = abs(principal_components[:, 0]) - abs(f_c_o)
    aux = [principal_components[:, 0], f_c_o, abs(errors)]
    f.write('First Component\n')
    aux = pd.DataFrame(data=np.transpose(aux), index=df.index.values,
                       columns=['Library', 'Oja', 'Error'])
    aux.sort_values(by='Library', axis=0, inplace=True)
    f.write(aux.to_string())

    f.close()

    f_c_o_data_frame = pd.DataFrame(data=f_c_o, index=df.index.values, columns=['Component 1'])
    f_c_o_data_frame.sort_values(by='Component 1', axis=0, inplace=True)
    SeaGraph.graph_barplot(f_c_o_data_frame.values[:, 0], f_c_o_data_frame.index.values, title="First PC per Country")

    SeaGraph.graph_barplot(results.w[-1], df.columns.values, title="Variables Loads")


if __name__ == '__main__':
    main()
