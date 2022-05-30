import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from algorithms.Kohonen import Kohonen
from utils import SeaGraph
from utils.Kohonen.ConfigULK import Config
from utils.Kohonen.KohonenParameters import KohonenParameters

sns.set_theme()


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

    SeaGraph.graph_boxplot(data, df.columns.values, title="Non Standardized Data")
    SeaGraph.graph_boxplot(standardize_data, df.columns.values, title="Standardized Data")

    parameters = KohonenParameters(config)
    kohonen = Kohonen(parameters, standardize_data)
    kohonen.train(standardize_data)
    kohonen_results = kohonen.get_results(standardize_data, df.index)

    SeaGraph.graph_heatmap(kohonen_results.elements_per_neuron, annot=kohonen_results.labels,
                           title="Associated Countries per Neuron\neta = " + str(config.learning_rate) + " | initial_r = " + str(config.initial_radius),
                           c_map="GnBu")
    SeaGraph.graph_heatmap(kohonen_results.weight_mean, annot=np.round(kohonen_results.weight_mean, decimals=4),
                           title="Mean Distance Between Neighbour Neurons",
                           c_map="Greys")


if __name__ == '__main__':
    main()
