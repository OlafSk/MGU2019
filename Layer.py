import numpy as np
def sigmoid(x):
    return np.exp(x) / np.sum(np.exp(x))

def relu(x):
    return np.where(x > 0, x, 0)

def b_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def b_relu(x):
    return np.where(x > 0, 1, 0)

FORWARD_FUNCTION_DICT = {
    "sigmoid": sigmoid,
    "relu": relu
}
BACKWARD_FUNCTION_DICT = {
    "sigmoid": b_sigmoid,
    "relu": b_relu
}

class Layer:
    """Single layer in multi-layer perceptron.

    Parameters
    ----------
    units: int
        Specifies number of neurons in this layer
    input_shape: int
        Specifies number of neurons in previous layer or input data
    activation_function: string ("relu", "sigmoid", "linear"):
        Specifies activation function to use as output of this layer.

    Attributes
    ----------
    W: np.array
        Matrix of weights in this layer
    B: np.array
        Bias array in this layer
    """

    def __init__(self, units, input_shape, activation_function):
        #Initialize all parameters to passed values
        self.units = units
        self.input_shape = input_shape
        self.forward_activation_function = FORWARD_FUNCTION_DICT[activation_function]
        self.backward_activation_function = BACKWARD_FUNCTION_DICT[activation_function]
        #Generate weight matrix with normal distribution
        self.W = np.random.normal(size = (self.input_shape, self.units))
        #initialize bias
        self.B = np.zeros(self.units) + 1
        #Generate derrivative matrix for weights and bias
        self.DW = np.zeros_like(self.W)
        self.DB = np.zeros_like(self.B)

    def forward_pass(self, X):
        self.Z = X @ self.W + self.B
        self.forward = self.forward_activation_function(X @ self.W + self.B)
        return self.forward

    def backward_pass(self, prev_G, X):
        DW = prev_G @ X.T
        DB = prev_G
        return DW, DB


layer = Layer(units = 10, input_shape = 2, activation_function = "sigmoid")

layer.forward_pass(np.random.normal(size=(3,2)))
