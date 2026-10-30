from .losses import Loss
import numpy as np

class NeuralNetwork:
    def __init__(self, layers=None):
        self.layers = list(layers) if layers is not None else []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_pass(self, x):
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x

    def backpropagation(self, loss_derivative):
        for layer in reversed(self.layers):
            loss_derivative = layer.backward_pass(loss_derivative)

    def train(self, X, y, epochs, optimizer, batch_size = None,shuffle = False):

        if len(X) == 0:
            raise ValueError("Empty data")

        if len(X) != len(y):
            raise ValueError("target and input size mismatched")
        
        if batch_size is not None:
            if not isinstance(batch_size,int) or isinstance(batch_size,bool):
                raise ValueError("batch size must be integer")
            elif batch_size <=0:
                raise ValueError("Batch size can't be negative")

        samples = np.shape(X)[0]

        if batch_size is None:
            batch_size = samples
            
        if samples % batch_size != 0:
            total_batches = (samples// batch_size) + 1
        else: 
            total_batches = samples// batch_size

        indices = np.arange(samples)
        

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(indices)
            batch_loss = 0
            for batch in range(total_batches):
                batch_indices = indices[batch * batch_size:(batch + 1) * batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                y_pred = self.forward_pass(batch_X)
                loss = Loss.MSE_Loss(y_pred, batch_y)

                batch_loss += (loss * len(batch_X))

                loss_derivative = Loss.MSE_derivative(y_pred, batch_y)
                self.backpropagation(loss_derivative)
                optimizer.step(self.layers)

            print(f"Loss At epoch {epoch} is {batch_loss/samples}")

    def predict(self, x):
        return self.forward_pass(x)
