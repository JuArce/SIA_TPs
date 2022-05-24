import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

variables = ['Area', 'GDP', 'Inflation', 'L.expect', 'Military', 'P.growth', 'Unemployment']
components = ['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6', 'Component 7']
eigenvectors_c = ['1', '2', '3', '4', '5', '6', '7']


def __main__():
    np.set_printoptions(suppress=True)
    print('Argument List:', str(sys.argv))
    print(sys.argv)
    assert len(sys.argv) == 2, 'Missing arguments'

    f = open("./resources/Results.txt", "w")

    df = pd.read_csv(sys.argv[1])
    countries = df.values[:, 0]
    df.set_index('Country', drop=True, inplace=True)
    data = df.values

    # Standardize the data
    data = StandardScaler().fit_transform(data)

    # Principal components
    pca = PCA()
    principal_components = pca.fit_transform(data)

    principal_df = pd.DataFrame(data=principal_components,
                                columns=components, index=df.index.values)
    f.write("------------------------------\n")
    f.write("Principal Components\n\n")
    f.write(principal_df.to_string())
    f.write("\n------------------------------\n\n")

    # # Eigenvectors
    f.write("Eigenvectors\n\n")
    eigenvectors = pd.DataFrame(data=np.transpose(pca.components_), columns=eigenvectors_c)
    f.write(eigenvectors.to_string())
    f.write("\n------------------------------\n\n")

    np.set_printoptions(suppress=True)

    # Mostrar que los vectores son ortogonales
    f.write("Ortogonal matrix\n\n")
    ortogonal_matrix = pd.DataFrame(data=np.round(np.transpose(pca.components_).dot(pca.components_), decimals=5))
    f.write(ortogonal_matrix.to_string())
    f.write("\n------------------------------\n\n")

    # Pesos primera componente
    f.write("First component weight per variable\n")
    f_c = pd.DataFrame(data=np.transpose(pca.components_)[:, 0], index=variables)
    f.write(f_c.to_string())
    f.write("\n------------------------------\n\n")

    # Primera componente de cada pais
    f.write("First component on each country\n")
    f_c = pd.DataFrame(data=principal_components[:, 0], index=df.index.values, columns=['Component 1'])
    f_c.sort_values(by='Component 1', axis=0, inplace=True)
    f.write(f_c.to_string())
    f.write("\n------------------------------\n\n")

    # Cuanta información tiene cada componente principal
    f.write("Variance ratio\n")
    vr = pd.DataFrame(data=pca.explained_variance_ratio_, index=components)
    f.write(vr.to_string())
    f.write("\n------------------------------\n\n")

    # Graficos
    x_country = principal_df.values[:, 0]
    y_country = principal_df.values[:, 1]
    x_scale = 1.0 / (x_country.max() - x_country.min())
    y_scale = 1.0 / (y_country.max() - y_country.min())
    plt.figure(figsize=(10, 10), dpi=400)
    for i in range(len(variables)):
        plt.arrow(0, 0, pca.components_[0][i], pca.components_[1][i], color='black', alpha=0.5)
        plt.text(pca.components_[0][i] * 1.05, pca.components_[1][i] * 1.05, variables[i], color='black', ha='center',
                 va='center')

    plt.plot(x_country * x_scale, y_country * y_scale, color='blue', marker='o', linestyle='none', markersize=2)
    for i in range(len(countries)):
        plt.text(x_country[i] * x_scale, y_country[i] * y_scale + 0.01, countries[i], color='teal', ha='center',
                 va='center', fontsize=6)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    plt.savefig("./resources/components")
    plt.clf()

    # pie chart
    explode = (0.05, 0, 0, 0, 0, 0, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(pca.explained_variance_ratio_, explode=explode, labels=range(1, len(pca.explained_variance_ratio_) + 1),
            autopct='%1.0f%%',
            textprops={'size': 'smaller'}, startangle=90, pctdistance=0.85)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig("./resources/variance_ratio")

    f.close()


if __name__ == "__main__":
    __main__()
