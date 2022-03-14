
def hamming(state: str, goal: str):
    h = 0
    for idx, s in enumerate(state):
        if s != goal[idx] and s != '0':
            h += 1
    return h

