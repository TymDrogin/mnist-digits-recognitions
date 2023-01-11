import numpy as np

class Layer:

    def __init__(self, inNodes, outNodes) -> None:
        self.layer = np.zeros((inNodes, outNodes))


