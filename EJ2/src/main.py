import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

variables = ['Area', 'GDP', 'Inflation', 'L.expect', 'Military', 'P.growth', 'Unemployment']


def __main__():
    np.set_printoptions(suppress=True)
    print('Argument List:', str(sys.argv))
    print(sys.argv)
    assert len(sys.argv) == 2, 'Missing arguments'

    df = pd.read_csv(sys.argv[1])
    countries = df.values[:, 0]
    df.set_index('Country', drop=True, inplace=True)
    data = df.values

    # Standardize the data
    data = StandardScaler().fit_transform(data)

    pca = PCA()
    principal_components = pca.fit_transform(data)
    components = []
    for i in range(len(data[0])):
        aux = 'Principal Component ' + str(i + 1)
        components.append(aux)

    principal_df = pd.DataFrame(data=principal_components,
                                columns=components)

    pca_components = pca.components_
    # Eigenvectors
    # print("Components")
    # print(pca_components)
    # print("------------------------------")

    # Mostrar que los vectores son ortogonales

    # print("Principal df")
    # print(principal_df)
    # print("------------------------------")

    # Pesos primera componente
    print("------------------------------")
    print("First component weight per variable")
    for i in range(len(variables)):
        print(variables[i] + ":   " + str(pca_components[0][i]))
    print("------------------------------")

    # Primera componente de cada pais
    print("First component on each country")
    for i in range(len(countries)):
        print(countries[i] + ":   " + str(principal_df.values[i][0]))
    print("------------------------------")

    # Cuanta información tiene cada componente principal
    print("Variance ratio")
    print(pca.explained_variance_ratio_)
    print("------------------------------")
    # print(data)

    # Graficos
    x_country = principal_df.values[:, 0]
    y_country = principal_df.values[:, 1]
    x_scale = 1.0 / (x_country.max() - x_country.min())
    y_scale = 1.0 / (y_country.max() - y_country.min())
    plt.figure(figsize=(10, 10), dpi=400)
    for i in range(len(variables)):
        plt.arrow(0, 0, pca_components[0][i], pca_components[1][i], color='black', alpha=0.5)
        plt.text(pca_components[0][i] * 1.05, pca_components[1][i] * 1.05, variables[i], color='black', ha='center',
                 va='center')

    plt.plot(x_country * x_scale, y_country * y_scale, color='blue', marker='o', linestyle='none', markersize=2)
    for i in range(len(countries)):
        plt.text(x_country[i] * x_scale, y_country[i] * y_scale + 0.01, countries[i], color='teal', ha='center',
                 va='center', fontsize=6)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    plt.savefig("./components")
    plt.clf()

    # pie chart
    explode = (0.05, 0, 0, 0, 0, 0, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(pca.explained_variance_ratio_, explode=explode, labels=range(1, len(pca.explained_variance_ratio_)+1), autopct='%1.0f%%',
            textprops={'size': 'smaller'}, startangle=90, pctdistance=0.85)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig("./variance_ratio")


if __name__ == "__main__":
    __main__()
