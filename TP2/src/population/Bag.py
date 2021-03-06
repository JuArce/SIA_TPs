import random
from datetime import datetime

from population.Element import Element
from utils.fitness import get_fitness

TRUE = '1'
FALSE = '0'


class Bag:

    def __init__(self, max_weight: int, total_items: int, population: int, elements: [Element]):
        self.max_weight: int = max_weight
        self.total_items: int = total_items
        self.elements: [Element] = elements
        self.population: int = population
        self.chromosomes: dict = self.initialize_chromosomes()
        self.evolution = dict()  # Key: Gen Value: Max fitness

    def initialize_chromosomes(self) -> dict:
        i: int = 0

        chromosomes: dict = dict()

        random.seed(datetime.now())

        while i < self.population:
            chromosome = []
            j: int = 0

            while j < self.total_items:
                p = random.random()
                if p < 0.1:
                    chromosome.append(TRUE)
                else:
                    chromosome.append(FALSE)
                j += 1

            chromosome = ''.join(chromosome)

            if chromosome not in chromosomes:
                chromosomes[chromosome] = get_fitness(chromosome, self.elements, self.max_weight)
                i += 1

        return chromosomes
