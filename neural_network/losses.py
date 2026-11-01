import numpy as np
from .activations import softmax

class Loss:
    @staticmethod
    def MSE_Loss(predicted, actual):
        return np.mean((actual - predicted) ** 2)

    @staticmethod
    def MSE_derivative(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size


class MeanSquaredError:

    @staticmethod
    def forward(predicted, actual):
        return np.mean((actual - predicted) ** 2)

    @staticmethod
    def backward(predictions, targets):
        return 2 * (predictions - targets) / targets.size


class SoftmaxCrossEntropy:

    @staticmethod
    def validate_inputs(logits, targets):

        # 1. Logits must be 2D: (N, C)
        if logits.ndim != 2:
            raise ValueError("Logits must have shape (N, C)")

        N, C = logits.shape

        # 2. N and C must be greater than 0
        if N == 0 or C == 0:
            raise ValueError("Logits cannot be empty")

        # 3. Targets must be 1D: (N,)
        if targets.ndim != 1:
            raise ValueError("Targets must have shape (N,)")

        # 4. Number of samples must match
        if targets.shape[0] != N:
            raise ValueError("Logits and targets sample count mismatch")

        # 5. Targets must contain integers
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Targets must be integers")

        # 6. Target class IDs must be in [0, C)
        if np.any(targets < 0) or np.any(targets >= C):
            raise ValueError("Target class index out of range")

    @staticmethod
    def forward(logits, targets):
        SoftmaxCrossEntropy.validate_inputs(logits, targets)
        shifted = logits - np.max(logits,axis=1,keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted),axis=1,keepdims=True))

        loss = -np.mean(
            log_probs[np.arange(logits.shape[0]), targets]
        )
        return loss

    @staticmethod
    def backward(logits, targets):
        SoftmaxCrossEntropy.validate_inputs(logits, targets)
        gradient  = softmax(logits.copy())
        gradient[np.arange(logits.shape[0]),targets] -= 1
        gradient /= logits.shape[0]
        return gradient


