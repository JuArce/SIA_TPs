import random


# The slicing starts with the start_pos index (included) and ends at end_pos index (excluded).
# The step parameter is used to specify the steps to take from start to end index.
def simple(chromosomes: [str]):
    p = random.choice(range(len(chromosomes[0]) - 1))
    output: [str] = [chromosomes[0][0: p] + chromosomes[1][p: len(chromosomes[1])],
                     chromosomes[1][0: p] + chromosomes[0][p: len(chromosomes[0])]]

    return output
