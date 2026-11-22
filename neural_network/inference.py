import numpy as np
from neural_network import DenseLayer

class InferenceModel:
    def __init__(self, model, batch_size, reuse_buffers= False):
        self.model = model
        self.batch_size = batch_size
        self.reuse_buffers = reuse_buffers

        if not self.model.layers:
            raise ValueError("Model must contain at least one layer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")

        for layer in self.model.layers:
            if not isinstance(layer, DenseLayer) or layer.activation_function not in ['relu', 'linear','NoActivation']:
                raise ValueError("Only DenseLayer and relu/linear/Noactivation function supported.")


        if self.model.layers[0].weight_matrix.ndim != 2:
            raise ValueError("Weight matrices must be two-dimensional")
        self.input_features = self.model.layers[0].weight_matrix.shape[0]
        current_input = self.input_features
        self.dtype = self.model.layers[0].weight_matrix.dtype
        if not np.issubdtype(self.dtype, np.floating):
            raise ValueError("Parameters must have a floating dtype")
        
        buffers = []
        for layer in self.model.layers:
            if layer.weight_matrix.shape != (current_input, layer.units):
                raise ValueError("Weight shape must match input features and layer units")
            if layer.bias.shape != (1, layer.units):
                raise ValueError("Bias shape must be (1, layer units)")
            if layer.weight_matrix.dtype != self.dtype or layer.bias.dtype != self.dtype:
                raise ValueError("All parameters must have the same floating dtype")

            if reuse_buffers:
                buffers.append(np.empty((batch_size,layer.units),dtype=self.dtype))

            current_input = layer.units

        if buffers:
            self.buffers = buffers
        


    def predict(self, inputs):

        if np.shape(inputs) != (self.batch_size,self.input_features):
            raise ValueError(f"Input shape mismatched, current input shape:- {np.shape(inputs)}, expected:- {(self.batch_size,self.input_features)}")
        elif self.dtype != inputs.dtype:
            raise ValueError(f"dtype is mismatched, expected:- {self.dtype}, have:- {inputs.dtype}")

        currentInput = inputs

        if not self.reuse_buffers:
            for layer in self.model.layers:
                currentInput = np.dot(currentInput, layer.weight_matrix) + layer.bias
                if layer.activation_function == 'relu':
                    currentInput = np.maximum(0,currentInput)

        else:
            for i,layer in enumerate(self.model.layers):
                buffer = self.buffers[i]
                np.dot(currentInput,layer.weight_matrix,out=buffer)
                buffer += layer.bias
                if layer.activation_function == 'relu':
                    np.maximum(0,buffer, out = buffer)
                currentInput = buffer

        return currentInput.copy() if self.reuse_buffers else currentInput
