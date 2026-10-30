import numpy as np

from neural_network import DenseLayer, Loss, NeuralNetwork


def numerical_gradient(loss_fn, values, epsilon=1e-5):
    gradient = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        original = values[index]
        try:
            values[index] = original + epsilon
            loss_plus = loss_fn()
            values[index] = original - epsilon
            loss_minus = loss_fn()
        finally:
            values[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2 * epsilon)
    return gradient


def compare_gradients(analytical, numerical):
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)
    return np.max(np.abs(analytical - numerical))


def check_network(name, layers, inputs, targets):
    model = NeuralNetwork(layers)
    predictions = model.forward_pass(inputs)
    gradient = Loss.MSE_derivative(predictions, targets)
    for layer in reversed(model.layers):
        gradient = layer.backward_pass(gradient)

    checks = [("inputs", inputs, gradient.copy())]
    for index, layer in enumerate(model.layers):
        checks.extend([
            (f"layer {index} weights", layer.weight_matrix, layer.dweights.copy()),
            (f"layer {index} bias", layer.bias, layer.dbias.copy()),
        ])
    originals = [values.copy() for _, values, _ in checks]

    def loss_fn():
        return Loss.MSE_Loss(model.forward_pass(inputs), targets)

    max_error = 0.0
    for label, values, analytical in checks:
        numerical = numerical_gradient(loss_fn, values)
        try:
            error = compare_gradients(analytical, numerical)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {label}\n{exc}") from exc
        max_error = max(max_error, error)

    for (_, values, _), original in zip(checks, originals):
        np.testing.assert_array_equal(values, original)
    model.forward_pass(inputs)
    print(f"PASS {name}: max absolute error {max_error:.2e}")


def check_failure_detection():
    try:
        compare_gradients(np.array([1.1]), np.array([1.0]))
    except AssertionError:
        print("PASS incorrect gradient is rejected")
    else:
        raise AssertionError("Incorrect gradient was accepted")


def main():
    np.random.seed(42)
    inputs = np.random.randn(4, 3)
    targets = np.random.randn(4, 2)

    for activation in ("linear", "tanh", "sigmoid", "relu"):
        layer = DenseLayer(3, 2, activation)
        if activation == "relu":
            # Exercise both branches, safely away from the kink at zero.
            layer.weight_matrix[:] = np.array([
                [0.5, -0.5], [0.25, -0.25], [-0.1, 0.1]
            ])
            layer.bias[:] = np.array([[2.0, -2.0]])
            z = inputs @ layer.weight_matrix + layer.bias
            assert np.min(np.abs(z)) > 0.1
            assert np.any(z > 0) and np.any(z < 0)
        check_network(activation, [layer], inputs, targets)

    check_network(
        "tanh -> linear",
        [DenseLayer(3, 4, "tanh"), DenseLayer(4, 2)],
        inputs,
        targets,
    )
    check_failure_detection()


if __name__ == "__main__":
    main()
