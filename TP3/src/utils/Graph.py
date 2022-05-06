import matplotlib.pyplot as plt


def graph(x=None, y=None, x_label='', y_label='', title='', points=None, points_color=None, output_dir=None):
    graph_init()
    if points is not None and points_color is not None:
        graph_points(points[:, 0], points[:, 1], points_color)
    if x is not None and y is not None:
        graph_plot(x, y)
    graph_description(x_label, y_label, title)
    if output_dir:
        graph_save(output_dir)
    else:
        graph_show()


def graph_init(width=7, height=7):
    plt.clf()
    plt.figure(figsize=(width, height), layout='constrained', dpi=200)


def graph_clear():
    plt.clf()


def graph_show():
    plt.show()


def graph_save(output_dir):
    plt.savefig(output_dir)


def graph_plot(x, y):
    plt.plot(x, y)


def graph_points(x, y, colors):
    plt.scatter(x, y, s=100, c=colors, label=colors)


def graph_description(x_label, y_label, title):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)




