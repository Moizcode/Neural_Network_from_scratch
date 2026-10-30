class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            layer.weight_matrix -= self.learning_rate * layer.dweights
            layer.bias -= self.learning_rate * layer.dbias