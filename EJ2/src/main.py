import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def __main__():
    np.set_printoptions(suppress=True)
    print('Argument List:', str(sys.argv))
    print(sys.argv)
    assert len(sys.argv) == 2, 'Missing arguments'

    df = pd.read_csv(sys.argv[1])
    df.set_index('Country', drop=True, inplace=True)
    data = df.values

    # Standardize the data
    data = StandardScaler().fit_transform(data)

    pca = PCA()
    principalComponents = pca.fit_transform(data)
    components = []
    for i in range(len(data[0])):
        aux = 'Principal Component ' + str(i + 1)
        components.append(aux)

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=components)

    # Eigenvectors
    print(pca.components_)

    #Mostrar que los vectores son ortogonales
    

    # print(principalDf)
    # Cuanta información tiene cada componente principal
    print(pca.explained_variance_ratio_)
    # print(data)


if __name__ == "__main__":
    __main__()
