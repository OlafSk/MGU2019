import numpy as np
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def relu(x):
    return np.where(x > 0, x, 0)

def linear(x):
    return x

def tanh(x):
    return 2/(1 + np.exp(-2 * x)) - 1

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def b_tanh(x):
    return 1 - tanh(x) ** 2

def b_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def b_linear(x):
    return np.zeros_like(x) + 1

def b_relu(x):
    return np.where(x > 0, 1, 0)

def b_softmax(x):
    return np.exp(np.sum(x)) / np.sum(np.exp(x)) ** 2

FORWARD_FUNCTION_DICT = {
    "sigmoid": sigmoid,
    "relu": relu,
    "linear": linear,
    "tanh": tanh,
    "softmax": softmax
}
BACKWARD_FUNCTION_DICT = {
    "sigmoid": b_sigmoid,
    "relu": b_relu,
    "linear" : b_linear,
    "tanh": b_tanh,
    "softmax": b_softmax
}
