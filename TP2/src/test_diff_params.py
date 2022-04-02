import copy
import os
import random
import sys
from typing import List

from utils.Config_ga import Config
from utils.graphs import *
from utils.Results_ga import Results
# Cross Over algorithms
from cross_over.multiple import multiple
from cross_over.simple import simple
from cross_over.uniform import uniform
from mutations.mutation import mutation
from population.Bag import Bag
from population.Element import Element
# Selection algorithms
from selection.boltzmann import boltzmann
from selection.elite import elite
from selection.rank import rank
from selection.roulette import roulette
from selection.tournament import tournament
from selection.truncated import truncated
from utils.Criteria import Criteria
from utils.fitness import get_fitness
from utils.selection_parameters import SelectionParameter

selection = {
    "boltzmann": boltzmann,
    "tournament": tournament,
    "truncated": truncated
}

cross_over = {
    "simple": simple,
}

POPULATION = 500

print('Argument List:', str(sys.argv))
assert len(sys.argv) == 4, 'Missing arguments'

config_dir = sys.argv[2]
charts_dir = sys.argv[3]

max_weight: int
total_items: int
elements: List[Element] = []

with open(sys.argv[1], 'r') as f:
    line = f.readline()
    count: int = 0

    while line:
        aux: List[str] = line.split()

        if count == 0:
            total_items = int(aux[0])
            max_weight = int(aux[1])
        else:
            aux: List[str] = line.split()
            element: Element = Element(int(aux[1]), int(aux[0]))
            elements.append(element)
        count += 1
        line = f.readline()

    f.close()

initial_bag: Bag = Bag(max_weight, total_items, int(POPULATION), elements)
results: ['Results'] = []

for root, dirs, files in os.walk(config_dir):
    for f in files:
        f = open(os.path.join(root, f))
        config: Config = Config(f.read())
        f.close()

        bag: Bag = copy.deepcopy(initial_bag)

        criteria: Criteria = Criteria(config, bag.chromosomes)
        selection_parameters: SelectionParameter = SelectionParameter(config)

        while not criteria.is_completed():
            new_gen: dict = dict()

            while len(new_gen) < bag.population:
                selected = random.sample([*bag.chromosomes.keys()], 2)
                children = cross_over[config.cross_over_algorithm](selected, config)

                for child in children:
                    child = mutation(child, config.mutation_probability)
                    if child not in new_gen and child not in bag.chromosomes:
                        new_gen[child] = get_fitness(child, bag.elements, bag.max_weight)
                    if len(new_gen) == bag.population:
                        break

            union = new_gen | bag.chromosomes
            bag.chromosomes = selection[config.selection_algorithm](union, selection_parameters)
            selection_parameters.current_gen += 1
            bag.evolution[selection_parameters.current_gen] = max(bag.chromosomes.values())
            criteria.update_criteria(bag.chromosomes)

        bag.chromosomes = dict(sorted(bag.chromosomes.items(), key=lambda item: item[1], reverse=True))
        results.append(Results(bag, config))

        # for chromosome in bag.chromosomes:
        #     weight = 0
        #     benefit = 0
        #     for i, value in enumerate(chromosome):
        #         weight += int(value) * elements[i].weight  # x_i * w_i
        #         benefit += int(value) * elements[i].value  # x_i * b_i
        #     print('Weight ' + weight.__str__() +
        #           ' | Benefit ' + benefit.__str__())

if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)

# Impresión de cada gráfico por algoritmo de selección
for s in selection.keys():
    get_charts_by_selection_algorithm(s, results, charts_dir)