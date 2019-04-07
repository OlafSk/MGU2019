import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork
import numpy as np
from losses import LOSSES
from activation_functions import *

y = np.array([[1,0,0]])
y_hat = np.array([[0.5,0.3, 0.2]])
LOSSES["cross_entropy"](y, y_hat)


nn = NeuralNetwork("cross_entropy", 0)
nn.add(Layer(1,2, "relu"))
nn.add(Layer(10,1, "sigmoid"))
nn.add(Layer(2,10, "softmax"))
nn.forward_pass(np.array([[1,2], [0,1]]))
nn.train(np.array([[1,2], [0,2]]), np.array([[1,2], [0,2]]))
nn.loss(np.array([[1,0], [0,1]]), nn.forward_pass(np.array([[1,2], [0,1]]))).sum()
nn.layers[-1].W
nn.layers[-1].DW
