import numpy as np
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def relu(x):
    return np.where(x > 0, x, 0)

def linear(x):
    return x

def b_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def b_linear(x):
    return np.zeros_like(x) + 1

def b_relu(x):
    return np.where(x > 0, 1, 0)

FORWARD_FUNCTION_DICT = {
    "sigmoid": sigmoid,
    "relu": relu,
    "linear": linear
}
BACKWARD_FUNCTION_DICT = {
    "sigmoid": b_sigmoid,
    "relu": b_relu,
    "linear" : b_linear
}
