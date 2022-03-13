def not_adm_heu(state: str, goal: str):
    h = 0

    for idx, s in enumerate(state):
        if s != goal[idx]:
            h += 8
    return h
