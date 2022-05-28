import seaborn as sns
from matplotlib import pyplot as plt


def graph_heatmap(data, annot=None, x_label='', y_label='', title='', c_map=None):
    plt.clf()
    sns.heatmap(data, annot=annot, fmt='', annot_kws={"size": 11}, linewidths=.5,
                cmap=c_map)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
