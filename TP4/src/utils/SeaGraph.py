import math

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_color_codes("pastel")


def graph_heatmap(data, annot=None, x_label='', y_label='', title='', c_map=None):
    plt.clf()
    sns.heatmap(data, annot=annot, fmt='', annot_kws={"size": 11}, linewidths=.5,
                cmap=c_map)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def graph_multi_heatmap(data, title=''):
    cols = 3
    rows = math.ceil((len(data)) / 3)
    plt.clf()
    fig, axes = plt.subplots(rows, cols)
    fig.suptitle(title)
    [axi.set_axis_off() for axi in axes.ravel()]
    for i in range(len(data)):
        row = math.floor(i / cols)
        col = i % cols
        sns.heatmap(ax=axes[row, col] if rows > 1 else axes[col], data=data[i],  linewidths=.5, linecolor='black', cmap="Blues",
                    yticklabels=False, xticklabels=False, cbar=False)

    plt.show()


def graph_barplot(x, y, title=''):
    plt.clf()
    plt.figure(figsize=(8, 8), layout='constrained', dpi=300)
    sns.barplot(x=x, y=y, color='b')
    plt.title(title)
    plt.show()