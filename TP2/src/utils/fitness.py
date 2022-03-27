from population.Element import Element


def get_fitness(chromosome: str, elements: [Element], max_weight: int):
    weight: int = 0
    benefit: int = 0

    for i, value in enumerate(chromosome):
        weight += int(value) * elements[i].weight  # x_i * w_i
        benefit += int(value) * elements[i].value  # x_i * b_i

    if weight > max_weight:
        weight = weight * (weight - max_weight)
        return benefit / weight
    return benefit
