import numpy as np

class Network:
    def __init__(self, layers_layout:list, activation:list) -> None:
        self.layers = []
        for i in range(len(layers_layout)-1):
            self.layers.append(Layer(layers_layout[i], layers_layout[i+1]))


    def activation_function(x, activation_name):
        if activation_name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif activation_name == "relu":
            return np.maximum(0, x)
        elif activation_name == "tanh":
            return np.tanh(x)
        else:
            raise ValueError(f"Invalid activation function: {activation_name}")
        

class Layer:
    #class represents one layer of the network

    def __init__(self, inNodes, outNodes) -> None:
        self.layer = np.random.rand(inNodes, outNodes)
        self.bias = np.random.rand(outNodes)

    def weighted_sum(self, vector):
        return np.dot(vector, self.layer) + self.bias




