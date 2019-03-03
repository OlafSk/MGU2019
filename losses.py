import numpy as np
def mean_squared_error(y_hat, y):
    return np.sum((y - y_hat) ** 2, axis=-1) * 0.5

def b_mean_squared_error(y_hat, y):
    return np.sum((y - y_hat), axis=-1) * 2


LOSSES = {
    "mse": mean_squared_error
}

B_LOSSES = {
    "mse": b_mean_squared_error
}
