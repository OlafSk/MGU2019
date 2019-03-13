from Layer import Layer
from NeuralNetwork import NeuralNetwork
import numpy as np
nn = NeuralNetwork(loss="mse", momentum=0.99)

#nn.add(Layer(units = 1, input_shape=10, activation_function="relu"))
nn.add(Layer(units = 5, input_shape=10, activation_function="sigmoid"))
nn.add(Layer(units = 5, input_shape=5, activation_function="sigmoid"))

nn.add(Layer(units = 1, input_shape=5, activation_function="linear"))

X = np.random.normal(size=(1,10))
y_hat = nn.forward_pass(X)

y = [[1]]

nn.train(X, y, epochs=100, verbose=False, learning_rate=1e-2)
nn.forward_pass(X)
nn.loss(y, nn.forward_pass(X)).sum()
nn.forward_pass(X)
