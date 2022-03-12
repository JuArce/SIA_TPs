
def levenshtein(state: str, goal: str):
    distance = [[0 for _ in range(len(state) + 1)] for _ in range(len(goal) + 1)]
    for i in range(len(state) + 1):
        distance[i][0] = i
    for i in range(len(goal) + 1):
        distance[0][i] = i
    for i in range(1, len(state) + 1):
        for j in range(1, len(goal) + 1):
            distance[i][j] = min(
                distance[i-1][j] + 1,
                distance[i][j-1] + 1,
                distance[i-1][j-1] + 1 if state[i-1] != goal[j-1] else distance[i-1][j-1]
            )
    return distance[len(state)][len(goal)]

