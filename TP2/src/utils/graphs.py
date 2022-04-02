import matplotlib.pyplot as plt

from TP2.src.utils.results import Results


def get_charts_by_selection_algorithm(selection_algorithm_selected: str, results: [Results], charts_dir: str):
    plt.clf()
    plt.figure(figsize=(7, 7), layout='constrained', dpi=200)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title("Evolución en cada generación")
    plt.grid(True)

    for r in results:
        if r.config.selection_algorithm == selection_algorithm_selected:
            plt.plot(r.bag.evolution.keys(), r.bag.evolution.values(), label=r.config.__str__())
    plt.legend(loc='lower right')
    plt.savefig(
        charts_dir + '/' + str(selection_algorithm_selected) + '_selection_algorithms'  '.png')


def get_charts_by_cross(cross_over_selected: str, results: [Results], charts_dir: str):
    plt.clf()
    plt.figure(figsize=(7, 7), layout='constrained', dpi=200)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title("Evolución en cada generación")
    plt.grid(True)

    for r in results:
        if r.config.cross_over_algorithm == cross_over_selected:
            plt.plot(r.bag.evolution.keys(), r.bag.evolution.values(), label=r.config.__str__())
    plt.legend(loc='lower right')
    plt.savefig(
        charts_dir + '/' + str(cross_over_selected) + '_selection_algorithms'  '.png')
