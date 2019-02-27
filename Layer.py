import numpy as np
from inspect import currentframe, getargvalues
def sigmoid(x):
    return np.exp(x) / np.sum(np.exp(x))

def relu(x):
    return np.where(x > 0, x, 0)


FORWARD_FUNCTION_DICT = {
    "sigmoid": sigmoid,
    "relu": relu
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
    """

    def __init__(self, units, input_shape, activation_function):
        #Initialize all parameters to passed values
        self.units = units
        self.input_shape = input_shape
        self.forward_activation_function = FORWARD_FUNCTION_DICT[activation_function]
        #Generate weight matrix with normal distribution
        self.W = np.random.normal(size = (self.input_shape, self.units))
        #initialize bias
        self.B = np.zeros(self.units) + 1
    def forward_pass(self, X):
        return self.forward_activation_function(X @ self.W + self.B)

    def backward_pass():
        pass



layer = Layer(units = 10, input_shape = 2, activation_function = "sigmoid")
layer.forward_pass(np.random.normal(size=(3,2)))
