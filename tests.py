from Layer import Layer
from NeuralNetwork import NeuralNetwork
import numpy as np
nn = NeuralNetwork(loss="mse")

#nn.add(Layer(units = 1, input_shape=10, activation_function="relu"))
nn.add(Layer(units = 5, input_shape=10, activation_function="linear"))
nn.add(Layer(units = 5, input_shape=5, activation_function="linear"))

nn.add(Layer(units = 1, input_shape=5, activation_function="linear"))

X = np.random.normal(size=(1,10))
y_hat = nn.forward_pass(X)
y_hat
y = [[1]]

nn.train(X, y, epochs=100, verbose=False, learning_rate=1e-10)
nn.loss(y, nn.forward_pass(X)).sum()
nn.forward_pass(X)
