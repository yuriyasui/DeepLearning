import numpy as np

def softmax(a):
    if a.ndim == 2:
        c = np.max(a, axis=1).reshape(a.shape[0], 1)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a, axis=1).reshape(a.shape[0], 1)
        y = exp_a / sum_exp_a
    else:
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
    return y
