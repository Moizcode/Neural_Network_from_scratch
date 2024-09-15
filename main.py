import numpy as np
from model import NeuralNetwork
from Dense_layer import DenseLayer


# Data
input_size = 20000
input_variables = 3
x1 = np.random.rand(input_size)
x2 = np.random.rand(input_size)
x3 = np.random.rand(input_size)

y_true = 5*x1 + 2.4*x2 + 7.897*x3 + \
    np.random.rand(input_size) * np.random.rand(input_size)
y_true = y_true.reshape(input_size, 1)

input_X = np.dstack((x1, x2, x3))
input_X = input_X.reshape(input_size, input_variables)


# Define NN
model = NeuralNetwork([
    DenseLayer(3, 8, 'relu'),
    DenseLayer(8, 14, 'relu'),
    DenseLayer(14, 6, 'relu'),
    DenseLayer(6, 1)
])

model.train(input_X, y_true, 10, 0.15)
