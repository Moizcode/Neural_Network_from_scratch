import numpy as np

from neural_network import DenseLayer, NeuralNetwork


def main():
    input_size = 20000
    x1 = np.random.rand(input_size)
    x2 = np.random.rand(input_size)
    x3 = np.random.rand(input_size)

    y_true = (
        5 * x1
        + 2.4 * x2
        + 7.897 * x3
        + np.random.rand(input_size) * np.random.rand(input_size)
    ).reshape(input_size, 1)
    input_X = np.column_stack((x1, x2, x3))

    model = NeuralNetwork([
        DenseLayer(3, 8, "relu"),
        DenseLayer(8, 14, "relu"),
        DenseLayer(14, 6, "relu"),
        DenseLayer(6, 1),
    ])
    model.train(input_X, y_true, epochs=100, learning_rate=0.01)


if __name__ == "__main__":
    main()
