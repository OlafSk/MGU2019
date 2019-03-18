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

    def single_forward_pass(self, X):
        for layer in self.layers[1:]:
            X = layer.forward_pass(X)
        return X

    def forward_pass(self, X):
        """
        Forward pass of X through layers
        """
        y = np.zeros((X.shape[0], self.layers[-1].units))
        for i in range(X.shape[0]):
            X_ = X[i, :].reshape(-1,1)
            y[i,:] = self.single_forward_pass(X_).reshape(1, self.layers[-1].units)
        return y

    def calculate_loss(self, X, y):
        """
        loss of the function
        """
        return self.loss(self.forward_pass(X), y)

    def calculate_b_loss(self, X, y):
        return self.b_loss(self.forward_pass(X), y)


    def backward_pass(self, X, y):
        "first computeted manualy, next computed in the loop"
        for i in range(X.shape[0]):
            X_ = X[i, :].reshape(-1,1)
            y_ = y[i, :]
            if y_.shape[0] > 1:
                y_ = y_.reshape(-1,1)
            self.single_forward_pass(X_)
            self.layers[0].forward = X_
            last_layer = self.layers[-1]
            if self.loss == "mse":
                last_layer.semi_grad = ((self.layers[-1].forward - y_) * self.layers[-1].backward_activation_function(self.layers[-1].Z)).sum(axis=1, keepdims=True)
            if self.loss == "cross_entropy":
                last_layer.semi_grad = (((1-y_)/(1-self.layers[-1].forward) - y_/self.layers[-1].forward) * self.layers[-1].backward_activation_function(self.layers[-1].Z)).sum(axis=1, keepdims=True)
            last_layer.DB += self.layers[-1].semi_grad / X.shape[0]
            last_layer.DW += np.dot(last_layer.semi_grad, self.layers[-2].forward.reshape(1, -1)) / X.shape[0]
            semi_grad = last_layer.semi_grad
            for i in range(2, len(self.layers)):
                deriv = self.layers[-i].backward_activation_function(self.layers[-i].Z)
                semi_grad = np.dot(self.layers[-i+1].W.reshape(self.layers[-i+1].W.shape[1], self.layers[-i+1].W.shape[0]), semi_grad) * deriv
                self.layers[-i].DB += (self.layers[-i].last_grad_B * self.momentum + (1-self.momentum) * semi_grad) / X.shape[0]
                self.layers[-i].DW += (self.layers[-i].last_grad_W * self.momentum + (1-self.momentum) * np.dot(semi_grad, self.layers[-i-1].forward.T)) / X.shape[0]


    def zero_gradients(self):
        for layer in self.layers:
            layer.DW = np.zeros_like(layer.DW)
            layer.DB = np.zeros_like(layer.DB)


    def train(self, X, y, X_test=None, y_test=None, epochs = 1, learning_rate = 0.01, momentum = 0.99, verbose=True):
        """
        Method to train the network
        Momentum is not implemented yet.
        """
        train_loss = [0] * epochs
        test_loss = [0] * epochs
        grad_norm = [0] * epochs
        for i in range(epochs):

            self.zero_gradients()
            self.backward_pass(X, y)
            for layer in self.layers:
                layer.W -= layer.DW * learning_rate
                grad_norm[i] += np.linalg.norm(layer.DW, ord="fro")
                layer.last_grad_W = layer.DW
                layer.B -= layer.DB * learning_rate
                layer.last_grad_B = layer.DB
            if verbose:
                print("\r%d" % i, end="")
            train_loss[i] = self.loss(y, self.forward_pass(X)).sum() / X.shape[0]
            if X_test is not None and y_test is not None:
                test_loss[i] =self.loss(y_test, self.forward_pass(X_test)).sum() / X_test.shape[0]

        return train_loss, test_loss, grad_norm
