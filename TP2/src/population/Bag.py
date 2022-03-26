from population.Element import Element
from datetime import datetime
import random

TRUE = '1'
FALSE = '0'


class Bag:

    def __init__(self, max_weight: int, total_items: int, population: int, elements: [Element]):
        self.max_weight: int = max_weight
        self.total_items: int = total_items
        self.elements: [Element] = elements
        self.population: int = population
        self.chromosomes: set[str] = self.initialize_chromosomes()

    def initialize_chromosomes(self):
        i: int = 0

        chromosomes: set[str] = set()

        random.seed(datetime.now())

        while i < self.population:
            chromosome = []
            j: int = 0

            while j < self.total_items:
                p = random.random()
                if p < 0.5:
                    chromosome.append('1')
                else:
                    chromosome.append('0')
                j += 1

            chromosome = ''.join(chromosome)

            if chromosome not in chromosomes:
                chromosomes.add(chromosome)
                i += 1

        return chromosomes
