import numpy as np


def linear(z):
    return z


def linear_derivative(z, output):
    return np.ones_like(z)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z, output):
    return (z > 0).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z, output):
    return 1 - np.square(output)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z, output):
    return output * (1 - output)


ACTIVATIONS = {
    "NoActivation": (linear, linear_derivative),
    "linear": (linear, linear_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
}
