import numpy as np
from tqdm import tqdm


mnist_config = {
    "layers":[784, 20, 20, 10],
    "activation":["relu","relu","relu","softmax"],
    "learning_rate": 0.0001,
    "epochs":150,
}

class Network:
    def __init__(self, config) -> None:
        self.layers = []
        for i in range(len(config["layers"]) - 1):
            self.layers.append(Layer(config["layers"][i], config["layers"][i+1], config["activation"][i])) 

        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]

    def train(self, train_data):
        for i in tqdm(range(self.epochs - 1), desc="Epochs", ncols=80, colour="WHITE", ascii=" >="):
            pass
                

    def back_propagation(self, input_vector):

        self.feed_forward(input_vector)





    def feed_forward(self, input_vector):
        for layer in self.layers:
            input_vector = layer.activation(layer.weighted_sum(input_vector))
        return input_vector

    def cross_entropy(prediction, truth_vec):
        return -np.sum(prediction * np.log(truth_vec))


class Layer:
    def __init__(self, inNodes, outNodes, activation_function) -> None:
        self.weights = np.random.rand(inNodes, outNodes)
        self.bias = np.random.rand(outNodes)
        self.get_activation(activation_function)
     


    def weighted_sum(self, input_vector):
        weighted_sum = np.dot(input_vector, self.weights) + self.bias
        output_vector = self.activation(weighted_sum)

        self.input = input_vector
        self.output = weighted_sum

        return output_vector

    def gradiient(self):
        pass



    def get_activation(self, name):
        if name == "sigmoid":
            self.activation =  lambda x: 1 / (1 + np.exp(-x))
            self.derivative =  lambda x: x * (1 - x)
        elif name == "relu":
            self.activation =  lambda x: np.maximum(0, x)
            self.derivative = lambda x: np.where(x > 0, 1, 0)
        elif name == "tanh":
            self.activation =  lambda x: np.tanh(x)
            self.derivative = lambda x: 1 - x ** 2
        elif name == "softmax":
            self.activation =  lambda x: np.exp(x) / np.sum(np.exp(x))









