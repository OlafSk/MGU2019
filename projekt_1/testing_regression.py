import numpy as np
import pandas as pd
import seaborn as sns
from NeuralNetwork import NeuralNetwork
from Layer import Layer

df = pd.read_csv("Regression//data.activation.test.1000.csv")

df['y'] = - df['y']
sns.scatterplot(df['x'], df['y'])
sns.scatterplot(df['x'], nn.forward_pass(X).reshape(-1,))

np.random.seed(1234)
X = df['x'].values.reshape(-1,1)
y = df['y'].values.reshape(-1,1)
nn = NeuralNetwork("mse", 0)
nn.add(Layer(1, input_shape=1, activation_function="relu"))
nn.add(Layer(10, input_shape=1, activation_function="relu"))
nn.add(Layer(1, input_shape=10, activation_function="linear"))

nn.backward_pass(X, y)
nn.train(X, y, epochs=10, learning_rate=1e-5, verbose=0)
((y - nn.forward_pass(X))**2).sum() / X.shape[0]

nn.layers[1].DW
X.shape
nn.layers[3].semi_grad
nn.layers[2]
nn.layers.__len__()
nn.calculate_loss(X, y).sum()
