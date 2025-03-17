import numpy as np
import random as rd

class Layer:
    def __init__(self, neurons, prev_neurons, activation_function):
        self.n_neurons = neurons
        self.prev_n_nodes = prev_neurons+1
        self.activation_function = activation_function

        np.random.seed(20)
        self.weight_matrix = np.random.rand(self.prev_n_nodes, neurons)

    def multiply(self, input_array):
        ret = np.matmul(input_array, self.weight_matrix)
        ret = self.activation_function(ret)
        return ret

class NeuralNetwork:
    def __init__(self, n_input, n_output, n_hiddenlayer, hidden_layers_size, hidden_layers_function, output_layer_function):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddenlayer = n_hiddenlayer
        self.hidden_layers_size = hidden_layers_size
        self.hidden_layers_function = hidden_layers_function
        self.output_layer_function = output_layer_function
        
        self.initiateLayers()
        
    def initiateLayers(self):
        self.layers = []
        self.layers.append(Layer(self.hidden_layers_size[0], self.n_input, self.hidden_layers_function[0]))
        for i in range(1, self.n_hiddenlayer):
            self.layers.append(Layer(self.hidden_layers_size[i], self.hidden_layers_size[i-1], self.hidden_layers_function[i]))
        self.layers.append(Layer(self.n_output, self.hidden_layers_size[-1], self.output_layer_function))
    
    def forward(self, input_array):
        for layer in self.layers:
            input_array = np.append(input_array, 1)
            input_array = layer.multiply(input_array)
        return input_array

input = np.array([1,2,3,4,5,6])

function = lambda x:x

neural = NeuralNetwork(n_input=len(input), n_output=7, n_hiddenlayer=5, hidden_layers_size=[4, 3, 1, 6, 4], hidden_layers_function=[function, function, function, function, function], output_layer_function=function)
print(neural.forward(input))