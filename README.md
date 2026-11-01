# Neural network from scratch

A small neural network built with NumPy. Supports dense layers, linear, ReLU,
tanh and sigmoid activations, mean squared error for regression, and stable
softmax cross-entropy for multiclass classification.

## Setup

```bash
python -m pip install -r requirements.txt
python main.py
python -m examples.classification
```

## Structure

```text
neural_network/
    activations.py    Activation functions and their derivatives
    layers.py         Dense layer, forward and backward passes
    losses.py         Regression and classification losses
    optimizers.py     SGD parameter updates
    model.py          Layer sequencing and training
main.py               Regression example
examples/             Classification example
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

Training supports full batches or mini-batches, with SGD updates after
backpropagation. Inputs have shape `(samples, features)`. Regression targets have
shape `(samples, outputs)`; classification targets are integer class IDs with
shape `(samples,)`. Classification uses a linear output layer and
`loss_function=SoftmaxCrossEntropy()`.
Add activations in `activations.py`, layers in `layers.py`, and losses in
`losses.py` as the project grows.

The old root modules now live inside `neural_network`. The dense layer's
`weigth_matrix` attribute has been renamed to `weight_matrix`.
