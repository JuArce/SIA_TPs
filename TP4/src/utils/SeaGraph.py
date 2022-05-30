import math

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_color_codes("pastel")


def graph_heatmap(data, annot=None, x_label='', y_label='', title='', c_map=None):
    plt.clf()
    sns.heatmap(data, annot=annot, fmt='', annot_kws={"size": 11}, linewidths=.5,
                cmap=c_map, yticklabels=False, xticklabels=False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def graph_multi_heatmap(data, title='', cols=3, size=8):
    rows = math.ceil((len(data)) / cols)
    plt.clf()
    fig, axes = plt.subplots(rows, cols, figsize=(size, 3 * rows))
    fig.suptitle(title)
    [axi.set_axis_off() for axi in axes.ravel()]
    for i in range(len(data)):
        row = math.floor(i / cols)
        col = i % cols
        sns.heatmap(ax=axes[row, col] if rows > 1 else axes[col], data=data[i], linewidths=.5, linecolor='black',
                    cmap="Blues",
                    yticklabels=False, xticklabels=False, cbar=False)

    plt.show()


def graph_barplot(x, y, title=''):
    plt.clf()
    plt.figure(figsize=(8, 8), layout='constrained', dpi=300)
    sns.barplot(x=x, y=y, color='b')
    plt.title(title)
    plt.show()


def graph_plot(x=None, y=None, x_label='', y_label='', title=''):
    plt.clf()
    plt.figure(figsize=(8, 8), layout='constrained', dpi=300)
    plt.plot(x, y)
    plt.scatter(x, y, s=50)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def graph_boxplot(data, labels, x_label='', y_label='', title=''):
    plt.clf()
    plt.figure(figsize=(8, 8), layout='constrained', dpi=300)
    ax = sns.boxplot(data=data)
    ax.set_xticklabels(labels)
    plt.xticks(rotation=45)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
