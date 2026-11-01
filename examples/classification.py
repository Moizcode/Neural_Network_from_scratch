import numpy as np

from neural_network import DenseLayer, NeuralNetwork, SGD, SoftmaxCrossEntropy


def make_dataset(seed=42):
    rng = np.random.default_rng(seed)
    train_x, train_y, test_x, test_y = [], [], [], []
    for label, center in enumerate(((-2.0, -2.0), (2.0, -2.0), (0.0, 2.0))):
        points = rng.normal(center, 0.4, size=(100, 2))
        train_x.extend(points[:80])
        train_y.extend([label] * 80)
        test_x.extend(points[80:])
        test_y.extend([label] * 20)

    train_x, train_y = np.array(train_x), np.array(train_y)
    test_x, test_y = np.array(test_x), np.array(test_y)
    order = rng.permutation(len(train_x))
    return train_x[order], train_y[order], test_x, test_y


def main():
    np.random.seed(42)
    train_x, train_y, test_x, test_y = make_dataset()
    model = NeuralNetwork([
        DenseLayer(2, 8, "tanh"),
        DenseLayer(8, 3),
    ])
    model.train(
        train_x, train_y, epochs=60, optimizer=SGD(0.05),
        loss_function=SoftmaxCrossEntropy(), batch_size=32, shuffle=True,
    )
    logits = model.predict(test_x)
    predicted_classes = np.argmax(logits, axis=1)
    accuracy = np.mean(predicted_classes == test_y)
    test_loss = SoftmaxCrossEntropy.forward(logits, test_y)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {accuracy:.1%} ({len(test_y)} held-out samples)")


if __name__ == "__main__":
    main()
