import numpy as np

def build_train(indexes: np.array, data_x: np.array, data_y: np.array, idx: int):
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    for i in range(len(indexes)):
        for j in indexes[i]:
            if i != idx:
                test_set_x.append(data_x[j])
                test_set_y.append(data_y[j])
            else:
                train_set_x.append(data_x[j])
                train_set_y.append(data_y[j])

    return np.array(train_set_x), np.array(train_set_y), np.array(test_set_x), np.array(test_set_y)
