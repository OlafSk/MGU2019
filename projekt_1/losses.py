import numpy as np
def mean_squared_error(y, y_hat):
    return 0.5 * ((y - y_hat) ** 2)

def b_mean_squared_error(y, y_hat):
    return (y - y_hat)

def cross_entropy_error(y, y_hat):
    return -(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

def b_cross_entropy_error(y, y_hat):
    return (1-y)/(1-y_hat) - y/y_hat

LOSSES = {
    "mse": mean_squared_error,
    "cross_entropy": cross_entropy_error
}

B_LOSSES = {
    "mse": b_mean_squared_error,
    "cross_entropy": b_cross_entropy_error
}
