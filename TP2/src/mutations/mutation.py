import random


def mutation(chromosome: str):
    output = []

    for i in range(len(chromosome)):
        p = random.random()
        # Hay que mutar si p < 0.05
        if p < 0.05:
            aux = "1" if chromosome[i] == '0' else "0"
            output.append(aux)
        else:
            output.append(chromosome[i])

    return "".join(output)
