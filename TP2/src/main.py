import random
from typing import List

from population.Element import Element
from population.Bag import Bag
from TP2.src.utils.Config import Config
from utils.fitness import get_fitness

# Selection algorithms
from selection.boltzmann import boltzmann
from selection.elite import elite
from selection.rank import rank
from selection.roulette import roulette
from selection.tournament import tournament
from selection.truncated import truncated

# Cross Over algorithms
from cross_over.multiple import multiple
from cross_over.simple import simple
from cross_over.uniform import uniform

from mutations.mutation import mutation

from utils.Criteria import Criteria

import sys

selection = {
    "boltzmann": boltzmann,
    "elite": elite,
    "rank": rank,
    "roulette": roulette,
    "tournament": tournament,
    "truncated": truncated
}

cross_over = {
    "multiple": multiple,
    "simple": simple,
    "uniform": uniform,
}

print('Argument List:', str(sys.argv))
assert len(sys.argv) == 3, 'Missing arguments'

config_file = open(sys.argv[2], 'r')
config: Config = Config(config_file.read())
config_file.close()

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
bag: Bag = Bag(max_weight, total_items, int(config.population), elements)

criteria: Criteria = Criteria(config.generations_quantity, config.limit_time, bag.chromosomes)

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
    bag.chromosomes = selection[config.selection_algorithm](union, config)
    criteria.update_criteria(bag.chromosomes)


bag.chromosomes = dict(sorted(bag.chromosomes.items(), key=lambda item: item[1], reverse=True))

for chromosome in bag.chromosomes:
    weight = 0
    benefit = 0
    for i, value in enumerate(chromosome):
        weight += int(value) * elements[i].weight  # x_i * w_i
        benefit += int(value) * elements[i].value  # x_i * b_i
    print('Weight ' + weight.__str__() +
          ' | Benefit ' + benefit.__str__())
