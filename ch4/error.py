import numpy as np

def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y ,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size