import numpy as np
from Dense_layer import DenseLayer
from Loss_function import Loss


class NeuralNetwork:
    def __init__(self,layers = []) -> None:
        self.layers = layers

    def add_layer(self,layer):
        self.layers.append(layer)
    
    def forward_pass(self,x):
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x
    
    def backpropagation(self,loss_derivative,learning_rate = 0.2):
        for layer in reversed(self.layers):
            loss_derivative = layer.backward_pass(loss_derivative,learning_rate)

    def train(self,X,y,epochs,learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward_pass(X)
            loss = Loss.MSE_Loss(y_pred,y)
            print(f"Loss At epoch {epoch} is {loss}")
            loss_derivative = Loss.MSE_derivative(y_pred,y)
            self.backpropagation(loss_derivative,learning_rate)

    def predict(self,x):
        return self.forward_pass(x)
