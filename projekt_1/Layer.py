import numpy as np
from activation_functions import *

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
        self.W = np.random.normal(size = (self.units, self.input_shape))
        #initialize bias
        self.B = np.zeros((self.units, 1)) + 1
        #Generate derrivative matrix for weights and bias
        self.DW = np.zeros_like(self.W)
        self.DB = np.zeros_like(self.B)
        self.last_grad_W = np.zeros_like(self.DW)
        self.last_grad_B = np.zeros_like(self.DB)


    def forward_pass(self, X):
        self.Z = np.dot(self.W, X) + self.B
        self.forward = self.forward_activation_function(np.dot(self.W, X) + self.B)
        return self.forward

    def backward_pass(self, prev_G, X):
        DW = prev_G @ X.T
        DB = prev_G
        return DW, DB


layer = Layer(units = 10, input_shape = 2, activation_function = "sigmoid")
layer.W
layer.forward_pass(np.random.normal(size=(2,1)))
