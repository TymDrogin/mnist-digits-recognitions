import numpy as np
from tqdm import tqdm



mnist_config = {
    "layers":[784, 20, 20, 10],
    "activaton":["relu","relu","relu","softmax"],
    "learning_rate": 0.0001,
    "epochs":150,
    "random_seed": 1,
    
}




class Network:
    def __init__(self, config) -> None:
        self.layers = []
        for i in range(len(config["layers"]) - 1):
            self.layers.append(Layer(config["layers"][i], config["layers"][i+1],config["activaton"][i])) #layer initialization with config values 

        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]

    def train(self, train_data):
        pass

    def predict(self):
        pass




class Layer:
    
    def __init__(self, inNodes, outNodes, activation_function) -> None:
        self.weights = np.random.rand(inNodes, outNodes)
        self.bias = np.random.rand(outNodes)
        self.activaton = Activation(activation_function)

    def feed_forward(self, input_vector):
        weighted_sum = np.dot(input_vector, self.weights) + self.bias
        output_vector = self.activaton(weighted_sum)
        return output_vector


    
        


class Activation:
    #return an objest with.activaton function and its derivative
    def __init__(self, function_type) -> None:
        self.get_activation_function(function_type)

    def get_activation_function(self, name):
        if name == "sigmoid":
            self.activaton =  lambda x: 1 / (1 + np.exp(-x))
            self.derivative = lambda x: x * (1 - x)
        elif name == "relu":
            self.activaton =  lambda x: np.maximum(0, x)
            self.derivative = lambda x: np.where(x > 0, 1, 0)
        elif name == "tanh":
            self.activaton =  lambda x: np.tanh(x)
            self.derivative = lambda x: 1 - x ** 2
        elif name == "softmax":
            self.activaton =  lambda x: np.exp(x) / np.sum(np.exp(x))


