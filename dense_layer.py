from nn.Module import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, shape):
        super().__init__()
        input_shape, output_shape = shape
        self._W = np.random.rand(input_shape, output_shape)
        self._b = np.zeros(output_shape)
        
        # previous inputs
        self._A = None
        
        self._dW = None
        self._db = None
        self._dA = None

    def forward(self, inputs):
        self._A = inputs
        return np.dot(inputs, self._W) + self._b

    def backward(self, prev_gradients):
        self._dW = np.dot(self._A.transpose(), prev_gradients)
        self._db = prev_gradients.sum(axis=0)
        self._dA = np.dot(prev_gradients, self._W.transpose())
        return self._dA
