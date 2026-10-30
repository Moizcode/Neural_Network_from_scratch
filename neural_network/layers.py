import numpy as np

from .activations import ACTIVATIONS


class DenseLayer:
    def __init__(self, inputs, units, activation_function="NoActivation"):
        if activation_function not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation_function}")

        self.units = units
        self.activation_function = activation_function
        self.activation, self.activation_derivative = ACTIVATIONS[activation_function]
        self.glorot_x = np.sqrt(6 / (inputs + units))
        self.weight_matrix = np.random.uniform(
            low=-self.glorot_x, high=self.glorot_x, size=(inputs, units)
        )
        self.bias = np.zeros((1, units))

    def forward_pass(self, inputs):
        self.input = inputs
        self.z = np.dot(inputs, self.weight_matrix) + self.bias
        self.a = self.activation(self.z)
        return self.a

    def backward_pass(self, dl_da):
        dl_dz = self.activation_loss(dl_da)
        self.dweights = np.dot(self.input.T, dl_dz)
        self.dbias = np.sum(dl_dz, axis=0, keepdims=True)
        dl_dx = np.dot(dl_dz, self.weight_matrix.T)

        return dl_dx

    def activation_loss(self, dl_da):
        return dl_da * self.activation_derivative(self.z, self.a)
