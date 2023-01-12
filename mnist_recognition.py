from argparse import ArgumentDefaultsHelpFormatter
import numpy as np




mnist_config = {
    "layers":[784, 20, 20, 10],
    "activation":["relu","relu","relu","softmax"],
    "learning_rate": 0.0001,
}

class Network:
    def __init__(self, config) -> None:
        self.layers = []
        for i in range(len(config["layers"]) -1):
            self.layers.append(Layer(config["layers"][i], config["layers"][i+1],config["activation"][i],)) #layer initialization with config values 

        self.learning_rate = config["learning_rate"]
        

class Layer:
    #class represents o ne layer of the network

    def __init__(self, inNodes, outNodes, activation_function) -> None:
        self.weights = np.random.rand(inNodes, outNodes)
        self.bias = np.random.rand(outNodes)
        self.activation = self.get_activation_function(activation_function)

    def get_activation_function(self, name):
        if name == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == "relu":
            return lambda x: np.maximum(0, x)
        elif name == "tanh":
            return lambda x: np.tanh(x)
        elif name == "softmax":
            return lambda x: np.exp(x) / np.sum(np.exp(x))
        else:
            raise ValueError(f"Invalid activation function: {name}")

    
    def forward(self, input_vector):
        weighted_sum = np.dot(input_vector, self.weights) + self.bias
        output_vector = self.activation(weighted_sum)
        return output_vector