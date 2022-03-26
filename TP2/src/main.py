from population.Element import Element
from population.Bag import Bag

import sys

print('Argument List:', str(sys.argv))
assert len(sys.argv) == 2, 'Missing config json'

max_weight: int
total_items: int
elements: list[Element] = []

with open(sys.argv[1], 'r') as f:
    line = f.readline()
    count: int = 0

    while line:
        aux: list[str] = line.split()

        if count == 0:
            total_items = int(aux[0])
            max_weight = int(aux[1])
        else:
            aux: list[str] = line.split()
            element: Element = Element(int(aux[1]), int(aux[0]))
            elements.append(element)
        count += 1
        line = f.readline()

    f.close()

    # FIXME: arreglar population
    bag: Bag = Bag(max_weight, total_items, 20, elements)
