# Neural network from scratch

A small neural network built with NumPy. Supports dense layers, linear, ReLU,
tanh and sigmoid activations, and mean squared error for regression.

## Setup

```bash
python -m pip install -r requirements.txt
python main.py
```

## Structure

```text
neural_network/
    activations.py    Activation functions and their derivatives
    layers.py         Dense layer, forward and backward passes
    losses.py         Mean squared error and its derivative
    model.py          Layer sequencing and training
main.py               Regression example
experiments/          Small standalone experiments
```

Import the main classes with:

```python
from neural_network import DenseLayer, NeuralNetwork

model = NeuralNetwork([
    DenseLayer(3, 8, "relu"),
    DenseLayer(8, 1),
])
```

Training uses full batches and updates weights during backpropagation. Inputs
have shape `(samples, features)` and targets have shape `(samples, outputs)`.
Add activations in `activations.py`, layers in `layers.py`, and losses in
`losses.py` as the project grows.

The old root modules now live inside `neural_network`. The dense layer's
`weigth_matrix` attribute has been renamed to `weight_matrix`.
