

class Layer:
    """
    Layer Base class
    """

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, prev_grads):
        pass

    def optimize(self):
        pass

