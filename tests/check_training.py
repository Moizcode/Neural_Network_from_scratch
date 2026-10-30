import contextlib
import copy
import io
import unittest

import numpy as np

from neural_network import DenseLayer, Loss, NeuralNetwork, SGD


def train_quietly(model, x, y, epochs=1, batch_size=None, shuffle=False, optimizer=None):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        model.train(
            x, y, epochs, optimizer if optimizer is not None else SGD(0.01),
            batch_size=batch_size, shuffle=shuffle,
        )
    return output.getvalue()


class CountingSGD(SGD):
    def __init__(self):
        super().__init__(0.01)
        self.calls = 0

    def step(self, layers):
        self.calls += 1
        super().step(layers)


class RecordingModel(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward_pass(self, x):
        self.batches.append(x.copy())
        return np.zeros((len(x), 1))

    def backpropagation(self, gradient):
        expected_targets = 2 * self.batches[-1] + 1
        expected_gradient = Loss.MSE_derivative(
            np.zeros_like(expected_targets), expected_targets
        )
        np.testing.assert_allclose(gradient, expected_gradient)


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.x = np.arange(10, dtype=np.float64).reshape(-1, 1) / 10
        self.y = 2 * self.x + 1
        self.model = NeuralNetwork([DenseLayer(1, 3, 'tanh'), DenseLayer(3, 1)])

    def assert_parameters_equal(self, first, second):
        for a, b in zip(first.layers, second.layers):
            np.testing.assert_allclose(a.weight_matrix, b.weight_matrix)
            np.testing.assert_allclose(a.bias, b.bias)

    def test_full_batch_equivalence(self):
        explicit = copy.deepcopy(self.model)
        train_quietly(self.model, self.x, self.y, epochs=3)
        train_quietly(explicit, self.x, self.y, epochs=3, batch_size=len(self.x))
        self.assert_parameters_equal(self.model, explicit)

    def test_partial_batch_matches_manual_updates(self):
        manual = copy.deepcopy(self.model)
        optimizer = CountingSGD()
        train_quietly(self.model, self.x, self.y, batch_size=4, optimizer=optimizer)
        self.assertEqual(optimizer.calls, 3)
        for start in (0, 4, 8):
            x_batch, y_batch = self.x[start:start + 4], self.y[start:start + 4]
            predictions = manual.forward_pass(x_batch)
            manual.backpropagation(Loss.MSE_derivative(predictions, y_batch))
            SGD(0.01).step(manual.layers)
        self.assert_parameters_equal(self.model, manual)

    def test_large_batch_has_one_update_per_epoch(self):
        optimizer = CountingSGD()
        train_quietly(self.model, self.x, self.y, epochs=3, batch_size=20, optimizer=optimizer)
        self.assertEqual(optimizer.calls, 3)

    def test_shuffling_preserves_pairs_and_inputs(self):
        model = RecordingModel()
        original_x, original_y = self.x.copy(), self.y.copy()
        train_quietly(model, self.x, self.y, epochs=2, batch_size=4, shuffle=True)
        self.assertEqual(len(model.batches), 6)
        epoch_orders = []
        for epoch in range(2):
            batches = model.batches[epoch * 3:(epoch + 1) * 3]
            self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
            order = np.concatenate(batches).ravel()
            np.testing.assert_array_equal(np.sort(order), self.x.ravel())
            epoch_orders.append(order)
        self.assertFalse(np.array_equal(epoch_orders[0], self.x.ravel()))
        self.assertFalse(np.array_equal(epoch_orders[0], epoch_orders[1]))
        np.testing.assert_array_equal(self.x, original_x)
        np.testing.assert_array_equal(self.y, original_y)

    def test_weighted_epoch_loss(self):
        model = RecordingModel()
        output = train_quietly(model, self.x, self.y, epochs=2, batch_size=4)
        lines = output.strip().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            reported = float(line.split(' is ')[-1])
            np.testing.assert_allclose(reported, np.mean(self.y ** 2))

    def test_invalid_batch_sizes(self):
        for batch_size in (0, -1, 2.5, 0.0, True, False, '4'):
            with self.subTest(batch_size=repr(batch_size)):
                with self.assertRaises(ValueError):
                    train_quietly(self.model, self.x, self.y, batch_size=batch_size)

    def test_invalid_datasets(self):
        cases = [(self.x[:0], self.y[:0]), (self.x, self.y[:5]), (self.x[:5], self.y)]
        for x, y in cases:
            with self.subTest(input_samples=len(x), target_samples=len(y)):
                with self.assertRaises(ValueError):
                    train_quietly(self.model, x, y)

    def test_linear_regression_learns(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(30, 3))
        y = x @ np.array([[2.0], [-1.0], [0.5]]) + 0.3
        model = NeuralNetwork([DenseLayer(3, 1)])
        before = Loss.MSE_Loss(model.predict(x), y)
        train_quietly(model, x, y, epochs=200, batch_size=8, shuffle=True)
        after = Loss.MSE_Loss(model.predict(x), y)
        self.assertLess(after, before * 0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
