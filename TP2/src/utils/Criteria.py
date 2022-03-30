from datetime import datetime
import copy
from typing import Optional

import math


class Criteria:

    def __init__(self,
                 generations_quantity: int,
                 limit_time: int,
                 current_generation: dict):
        # Time
        self.limit_time = limit_time  # cuánto tiempo tiene que correr el algoritmo genético
        self.initial_time = datetime.now()

        # Cantidad de generacioens
        self.limit_gen_quantity = generations_quantity  # hasta cuántas generaciones hay que analizar.
        self.current_gen_quantity = 0  # contiene que número de generación se está analizando.

        # Generaciones
        self.current_generation: dict = copy.deepcopy(current_generation)
        self.last_generation: Optional[dict] = None  # contiene la generación anterior.

        # Estructura: Una parte relevante de la población no cambia en una cantidad de generaciones.
        self.unchanged_percentage = 0.9  # Definir por parámetro TODO
        self.max_unchanged_generations = 100  # Definir por parámetro TODO
        self.unchanged_generations = 0

        # Contenido: El mejor fitness no cambia en una cantidad de generaciones.
        self.max_unchanged_fitness_generations = 50  # Definir por parámetro TODO La cantidad Máxima de generaciones que no pueden cambiar
        self.unchanged_fitness_generations = 0  # La cantidad de generaciones que no cambiaron hasta ahora
        self.max_fitness = 0  # El maximo fitness hasta ahora

    def is_completed(self):
        return self.time_is_done() or \
               self.generations_quantity_is_done() or \
               self.content_fitness_is_done() or \
               self.generation_structure_is_done()

    # Chequeos

    def time_is_done(self):
        return (datetime.now() - self.initial_time).total_seconds() > self.limit_time

    def generations_quantity_is_done(self):
        return self.current_gen_quantity >= self.limit_gen_quantity

    def generation_structure_is_done(self):
        return self.unchanged_generations >= self.max_unchanged_generations

    def content_fitness_is_done(self):
        return self.unchanged_fitness_generations >= self.max_unchanged_fitness_generations

    # Updateo de criterios

    def update_criteria(self, chromosomes: dict):
        self.current_gen_quantity += 1
        self.update_generations(chromosomes)
        self.update_structure()
        self.update_fitness()

    def update_generations(self, chromosomes: dict):
        self.last_generation = copy.deepcopy(self.current_generation)
        self.current_generation = copy.deepcopy(chromosomes)

    def update_structure(self):
        counter: int = 0
        for key in self.current_generation:
            if key in self.last_generation:
                counter += 1

        if (counter / len(self.current_generation)) >= self.unchanged_percentage:
            self.unchanged_generations += 1
        else:
            self.unchanged_generations = 0

    def update_fitness(self):
        max_local_fitness = max(self.current_generation.values())
        if math.isclose(self.max_fitness, max_local_fitness):
            self.unchanged_fitness_generations += 1
        else:
            self.unchanged_fitness_generations = 0
        self.max_fitness = max([self.max_fitness, max_local_fitness])
