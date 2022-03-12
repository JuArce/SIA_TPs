
def manhattan(state: str, goal: str):
    h = 0
    for i in state:
        if i != '0':
            dx = abs(state.index(i) % 3 - goal.index(i) % 3)
            dy = abs(state.index(i) // 3 - goal.index(i) // 3)
            h += (dx + dy)
    return h
