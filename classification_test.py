import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork
import numpy as np
from losses import LOSSES
from activation_functions import *

y = np.array([[1,0,0]])
y_hat = np.array([[0.5,0.3, 0.2]])
LOSSES["cross_entropy"](y, y_hat)


nn = NeuralNetwork("cross_entropy", 0.9)
nn.add(Layer(2,1, "softmax"))

nn.train(np.array([[1], [2]]), np.array([[0,1],[1,0]]))
nn.forward_pass(np.array([[1], [2]]))
