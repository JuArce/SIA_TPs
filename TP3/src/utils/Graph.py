import matplotlib.pyplot as plt


def graph(x=None, y=None, x_label='', y_label='', title='', points=None, points_color=None, e=None, output_dir=None,
          cell_text=None, rows=None, columns=None):
    graph_init()

    if points is not None and points_color is not None:
        if e is not None:
            graph_points_with_std_dev(points[:, 0], points[:, 1], e, points_color)
        else:
            graph_points(points[:, 0], points[:, 1], points_color)

    if x is not None and y is not None:
        graph_plot(x, y)

    if cell_text is not None:
        graph_table_aux(cell_text, rows, columns)

    graph_description(x_label, y_label, title)

    if output_dir:
        graph_save(output_dir)
    else:
        graph_show()


def graph_table(cell_text=None, rows=None, columns=None, output_dir=None):
    graph_clear()
    plt.figure(dpi=400)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    plt.subplots_adjust(left=0.2, bottom=0.8)

    if cell_text is not None:
        graph_table_aux(cell_text, rows, columns)

    if output_dir:
        graph_save(output_dir)
    else:
        graph_show()


def graph_init(width=7, height=7):
    plt.clf()
    plt.figure(figsize=(width, height), layout='constrained', dpi=300)


def graph_clear():
    plt.clf()


def graph_show():
    plt.show()


def graph_save(output_dir):
    plt.savefig(output_dir)


def graph_plot(x, y):
    plt.plot(x, y)


def graph_points(x, y, colors):
    plt.scatter(x, y, s=50, c=colors, label=colors)


def graph_points_with_std_dev(x, y, e, colors):
    plt.scatter(x, y, s=50, c=colors, label=colors)
    plt.errorbar(x, y, e, linestyle='None', ecolor='#000000')


def graph_description(x_label, y_label, title):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)


def graph_table_aux(cell_text, rows, columns):
    table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      loc='bottom')
    table.scale(1, 1.5)
