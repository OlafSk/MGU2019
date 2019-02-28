import numpy as np
from Layer import Layer
def mean_squared_error(y_hat, y):
    return np.sum((y - y_hat) ** 2, axis=-1) * 0.5

def b_mean_squared_error(y_hat, y):
    return np.sum((y - y_hat), axis=-1) * 2


LOSSES = {
    "mse": mean_squared_error
}

B_LOSSES = {
    "mse": b_mean_squared_error
}


class NeuralNetwork:
    def __init__(self, loss):
        #Initialize list of layers to empty list
        aritificial_layer = Layer(0,0, "relu")
        self.layers = [aritificial_layer]
        self.loss = LOSSES[loss]
        self.b_loss = B_LOSSES[loss]

    def add(self, layer):
        """
        add layer to model
        """
        assert isinstance(layer, Layer)
        self.layers.append(layer)

    def forward_pass(self, X):
        """
        forward pass of data through layers
        """
        X = X.copy()
        for layer in self.layers[1:]:
            X = layer.forward_pass(X)
        return X

    def calculate_loss(self, X, y):
        """
        loss of the function
        """
        return self.loss(self.forward_pass(X), y)

    def calculate_b_loss(self, X, y):
        return self.b_loss(self.forward_pass(X), y)

    def backward_pass(self, X, y):
        "first computeted manualy, next computed in the loop"
        self.layers[0].forward = X
        last_layer = self.layers[-1]
        last_layer.DW = self.calculate_b_loss(X, y) * self.layers[-2].forward.T * self.layers[-1].backward_activation_function(self.layers[-1].forward)
        last_layer.DB = self.calculate_b_loss(X, y) * self.layers[-1].backward_activation_function(self.layers[-1].forward)
        for i in range(len(self.layers[1:-1]), 0, -1):
            next_layer = self.layers[i+1]
            layer = self.layers[i]
            layer.DW = next_layer.W.T @ next_layer.DW * layer.backward_activation_function(layer.Z)
            layer.DB = layer.DW.sum(axis=-1)
        return self.layers[1].DW
