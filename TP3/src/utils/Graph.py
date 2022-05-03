import matplotlib.pyplot as plt


def graph(x, y, x_label, y_label, title, results=None, output_dir=None):
    plt.clf()
    plt.figure(figsize=(7, 7), layout='constrained', dpi=200)
    if results:
        plt.scatter(results.x[:, 0], results.x[:, 1], s=100, c=results.y)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    # plt.legend()
    if output_dir:
        plt.savefig(output_dir)
    else:
        plt.show()
