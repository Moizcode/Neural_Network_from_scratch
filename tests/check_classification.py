import contextlib
import io
import unittest
import warnings

import numpy as np

from neural_network import DenseLayer, MeanSquaredError, NeuralNetwork, SGD, SoftmaxCrossEntropy
from neural_network.activations import softmax
from tests.check_gradients import numerical_gradient


class ClassificationChecks(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.logits = np.array([[0.2, -0.4, 1.1], [-0.7, 0.8, 0.3]])
        self.targets = np.array([2, 0])
        self.loss = SoftmaxCrossEntropy()

    def test_softmax_values_and_normalization(self):
        probabilities = softmax(np.array([[2., 1., 0.], [0., 0., 0.]]))
        np.testing.assert_allclose(probabilities[0], [0.66524096, 0.24472847, 0.09003057])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertTrue(np.all(probabilities >= 0))

    def test_large_logits_are_stable(self):
        logits = np.array([[1000., -1000., 0.], [-1000., 1000., 0.]])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            probabilities = softmax(logits)
            loss = self.loss.forward(logits[:1], np.array([1]))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(loss, 2000)

    def test_shift_invariance(self):
        shifted = self.logits + np.array([[1000.], [-1000.]])
        np.testing.assert_allclose(softmax(self.logits), softmax(shifted))
        np.testing.assert_allclose(
            self.loss.forward(self.logits, self.targets),
            self.loss.forward(shifted, self.targets),
        )
        np.testing.assert_allclose(
            self.loss.backward(self.logits, self.targets),
            self.loss.backward(shifted, self.targets),
        )

    def test_uniform_logits_loss(self):
        np.testing.assert_allclose(
            self.loss.forward(np.zeros((2, 3)), self.targets), np.log(3)
        )

    def test_logits_gradient(self):
        numerical = numerical_gradient(
            lambda: self.loss.forward(self.logits, self.targets), self.logits
        )
        analytical = self.loss.backward(self.logits, self.targets)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(analytical.sum(axis=1), 0, atol=1e-12)

    def test_two_layer_parameter_gradients(self):
        inputs = np.random.randn(4, 2)
        targets = np.array([0, 1, 2, 0])
        model = NeuralNetwork([DenseLayer(2, 4, 'tanh'), DenseLayer(4, 3)])
        logits = model.forward_pass(inputs)
        gradient = self.loss.backward(logits, targets)
        for layer in reversed(model.layers):
            gradient = layer.backward_pass(gradient)
        checks = [(inputs, gradient.copy())]
        for layer in model.layers:
            checks.extend([
                (layer.weight_matrix, layer.dweights.copy()),
                (layer.bias, layer.dbias.copy()),
            ])
        for index, (values, analytical) in enumerate(checks):
            with self.subTest(array=index):
                numerical = numerical_gradient(
                    lambda: self.loss.forward(model.forward_pass(inputs), targets), values
                )
                np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)

    def test_inputs_are_unchanged(self):
        logits, targets = self.logits.copy(), self.targets.copy()
        softmax(self.logits)
        self.loss.forward(self.logits, self.targets)
        self.loss.backward(self.logits, self.targets)
        np.testing.assert_array_equal(self.logits, logits)
        np.testing.assert_array_equal(self.targets, targets)

    def test_invalid_loss_inputs(self):
        cases = [
            ('empty samples', np.empty((0, 3)), np.array([], dtype=int)),
            ('empty classes', np.empty((2, 0)), np.array([0, 0])),
            ('1D logits', np.zeros(3), np.array([0])),
            ('3D logits', np.zeros((2, 1, 3)), np.array([0, 1])),
            ('column targets', self.logits, np.array([[0], [1]])),
            ('float targets', self.logits, np.array([0., 1.])),
            ('boolean targets', self.logits, np.array([False, True])),
            ('short targets', self.logits, np.array([0])),
            ('extra targets', self.logits, np.array([0, 1, 2])),
            ('negative class', self.logits, np.array([-1, 0])),
            ('class out of range', self.logits, np.array([3, 0])),
        ]
        problems = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for method in (self.loss.forward, self.loss.backward):
                for name, logits, targets in cases:
                    try:
                        method(logits, targets)
                    except ValueError:
                        continue
                    except Exception as exc:
                        problems.append(f'{method.__name__}: {name} raised {type(exc).__name__}')
                    else:
                        problems.append(f'{method.__name__}: {name} was accepted')
        self.assertFalse(problems, '\n' + '\n'.join(problems))

    def test_mse_object_matches_legacy_loss(self):
        from neural_network import Loss
        np.testing.assert_allclose(
            MeanSquaredError.forward(self.logits, self.logits / 2),
            Loss.MSE_Loss(self.logits, self.logits / 2),
        )
        np.testing.assert_allclose(
            MeanSquaredError.backward(self.logits, self.logits / 2),
            Loss.MSE_derivative(self.logits, self.logits / 2),
        )

    def test_classifier_learns_held_out_clusters(self):
        rng = np.random.default_rng(42)
        train_x, train_y, test_x, test_y = [], [], [], []
        for label, center in enumerate(((-2., -2.), (2., -2.), (0., 2.))):
            points = rng.normal(center, 0.3, size=(30, 2))
            train_x.extend(points[:20])
            test_x.extend(points[20:])
            train_y.extend([label] * 20)
            test_y.extend([label] * 10)
        train_x, train_y = np.array(train_x), np.array(train_y)
        model = NeuralNetwork([DenseLayer(2, 8, 'tanh'), DenseLayer(8, 3)])
        initial = self.loss.forward(model.predict(train_x), train_y)
        with contextlib.redirect_stdout(io.StringIO()):
            model.train(
                train_x, train_y, epochs=150, optimizer=SGD(0.05),
                loss_function=self.loss, batch_size=16, shuffle=True,
            )
        final = self.loss.forward(model.predict(train_x), train_y)
        self.assertTrue(np.isfinite(final))
        self.assertLess(final, initial)
        predictions = model.predict(np.array(test_x)).argmax(axis=1)
        self.assertGreaterEqual(np.mean(predictions == np.array(test_y)), 0.9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
