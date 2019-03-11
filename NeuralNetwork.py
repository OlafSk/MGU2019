import numpy as np
from Layer import Layer
from losses import *


class NeuralNetwork:
    """Multi layer network.

    Parameters
    ----------
    loss: string ("mean_squared_error")
        Specifies loss to train the function on

    Attributes
    ----------
    layers: list
        list contaning all layers within network.
    """

    def __init__(self, loss, momentum):
        #Initialize list of layers to empty list
        aritificial_layer = Layer(0,0, "relu")
        self.layers = [aritificial_layer]
        self.loss = LOSSES[loss]
        self.b_loss = B_LOSSES[loss]
        self.momentum = momentum

    def add(self, layer):
        """
        Function used to add layer to network:
        Parameters
        ----------
        layer: Layer,
            layer which will be added to network.
        Returns
        -------
        None
        """
        assert isinstance(layer, Layer)
        self.layers.append(layer)

    def forward_pass(self, X):
        """
        Forward pass of X through layers
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
        last_layer.DW = (self.calculate_b_loss(X, y) * self.layers[-2].forward.T @
                    self.layers[-1].backward_activation_function(self.layers[-1].forward) /
                    X.shape[0])
        last_layer.DB = (self.calculate_b_loss(X, y) @
                    self.layers[-1].backward_activation_function(self.layers[-1].forward) /
                    X.shape[0]).sum(axis=0, keepdims=True)
        for i in range(len(self.layers[1:-1]), 0, -1):
            next_layer = self.layers[i+1]
            prev_layer = self.layers[i-1]
            layer = self.layers[i]
            layer.DW = self.momentum * layer.DW + (1-self.momentum) * (next_layer.W.T.sum(axis=0, keepdims=True) @ next_layer.DW.sum(axis=1, keepdims=True) * prev_layer.forward.T @
                        layer.backward_activation_function(layer.Z)  / X.shape[0])
            layer.DB = self.momentum * layer.DB + (1-self.momentum)*(next_layer.W.T.sum(axis=0, keepdims=True) @ next_layer.DW.sum(axis=1, keepdims=True) *
                        layer.backward_activation_function(layer.Z) / X.shape[0]).sum(axis=0, keepdims=True)

    def train(self, X, y, epochs = 1, learning_rate = 0.01, momentum = 0.99, verbose=True):
        """
        Method to train the network
        Momentum is not implemented yet.
        """
        for i in range(epochs):
            self.backward_pass(X, y)
            for layer in self.layers:
                layer.W += layer.DW * learning_rate
                layer.B += layer.DB * learning_rate
            if verbose:
                print(i)
