import random


def mutation(chromosome: str, mutation_probability):
    output = []

    for i in range(len(chromosome)):
        p = random.random()
        # Hay que mutar si p < 0.05
        if p < mutation_probability:
            aux = "1" if chromosome[i] == '0' else "0"
            output.append(aux)
        else:
            output.append(chromosome[i])

    return "".join(output)
