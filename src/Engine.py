import numpy as np
import random as rd
import pickle

class Layer:
    def __init__(self, neurons, prev_neurons, activation_function, bias, init_type="random-uniform"):
        self.n_neurons = neurons
        self.prev_n_nodes = prev_neurons+1
        self.activation_function = activation_function
        param1, param2, init_type = init_type

        if init_type=="random-uniform":
            self.initiateWeightRDUniform(lower_bound=param1, upper_bound=param2)
        elif init_type=="random-normal":
            self.initiateWeightRDNormal(mean=param1, variance=param2)
        elif init_type=="xavier":
            self.initiateWeightXavier()
        elif init_type=="he":
            self.initateWeightHe()
        elif init_type=="zero":
            self.initiateWeightZero()
        else:
            raise ValueError("Invalid intiation type")
        
        self.weight_matrix[-1, :] = bias
        self.weight_matrix.astype(np.float64)

        self.output = None
        self.delta = None

    def multiply(self, input_array):
        ret = np.matmul(input_array, self.weight_matrix)
        ret = ret.astype(np.float64)
        self.output = self.activation_function(ret)
        return self.output
    
    def update_weights(self, input_array, learning_rate):
        """ Update weights using backpropagation gradient descent """
        gradient = np.outer(input_array, self.delta)
        self.weight_matrix -= learning_rate * gradient
        
    def initiateWeightRDUniform(self, seed=0, lower_bound=0, upper_bound=1):
        np.random.seed(seed)
        self.weight_matrix = np.random.uniform(lower_bound, upper_bound, (self.prev_n_nodes, self.n_neurons))
    
    def initiateWeightRDNormal(self, mean, variance, seed):
        np.random.seed(seed)
        self.weight_matrix = np.random.normal(mean, variance, (self.prev_n_nodes, self.n_neurons))
    
    def initiateWeightZero(self):
        self.weight_matrix = np.zeros((self.prev_n_nodes, self.n_neurons))

    def initiateWeightXavier(self, seed=0):
        np.random.seed(seed)
        fan_avg = (self.prev_n_nodes-1 + self.n_neurons)/2
        variance = 1/fan_avg
        self.weight_matrix = np.random.normal(0, variance, (self.prev_n_nodes, self.n_neurons))
    
    def initateWeightHe(self, seed=0):
        np.random.seed(seed)
        variance = 2/(self.prev_n_nodes-1)
        self.weight_matrix = np.random.normal(0, variance, (self.prev_n_nodes, self.n_neurons))

class NeuralNetwork:
    def __init__(self, n_input, n_output, n_hiddenlayer, hidden_layers_size, hidden_layers_function, output_layer_function, bias, init_type):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddenlayer = n_hiddenlayer
        self.hidden_layers_size = hidden_layers_size
        self.hidden_layers_function = hidden_layers_function
        self.output_layer_function = output_layer_function
        self.init_type = init_type
        
        self.initiateLayers(bias)
        
    def initiateLayers(self, bias):
        self.layers = []
        self.layers.append(Layer(self.hidden_layers_size[0], self.n_input, self.hidden_layers_function[0], bias, self.init_type))
        for i in range(1, self.n_hiddenlayer):
            self.layers.append(Layer(self.hidden_layers_size[i], self.hidden_layers_size[i-1], self.hidden_layers_function[i], bias, self.init_type))
        self.layers.append(Layer(self.n_output, self.hidden_layers_size[-1], self.output_layer_function, bias, self.init_type))
    
    def forward(self, input_array):
        for layer in self.layers:
            input_array = np.append(input_array, 1)
            input_array = layer.multiply(input_array)
        return input_array
    
    def forwardBatch(self, input_matrix):
        for layer in self.layers:
            input_matrix = np.append(input_matrix, np.ones((input_matrix.shape[0], 1)), axis=1)
            input_matrix = layer.multiply(input_matrix)
        return input_matrix
    
    def backward(self, expected_output, learning_rate):
        """ Backpropagation algorithm """
        # Compute error at output layer
        last_layer = self.layers[-1]
        last_layer.delta = (last_layer.output - expected_output) * self.output_layer_function(last_layer.z, derivative=True)

        # Propagate error backwards
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            # Compute delta for hidden layers
            error_term = np.dot(next_layer.weight_matrix[:-1], next_layer.delta)
            layer.delta = error_term * self.hidden_layers_function[i](layer.z, derivative=True)

        # Update weightswadwd
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def train(self, inputs, expected_outputs, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                self.forward(inputs[i])
                self.backward(expected_outputs[i], learning_rate)

            if epoch % 1 == 0:
                loss = np.mean((expected_outputs - self.forward(inputs)) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.5f}")
    
    def trainBatch(self, inputs, expected_outputs, epochs, learning_rate):
        counter = 0
        while(True):
            res = self.forwardBatch(inputs)
            print(res)
                
class Engine():
    def __init__(self,
                 n_hiddenlayer,
                 hidden_layers_size,
                 hidden_layers_function,
                 output_layer_function,
                 bias,
                 init_type,
                 data_train,
                 data_train_class,
                 learning_rate,
                 epochs,
                 batch_size):
        self.data_train = data_train
        self.data_train_class = data_train_class
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.neural = NeuralNetwork(n_input=data_train.shape[1],
                                    n_output=np.unique(data_train_class).shape[0],
                                    n_hiddenlayer=n_hiddenlayer,
                                    hidden_layers_size=hidden_layers_size,
                                    hidden_layers_function=hidden_layers_function,
                                    output_layer_function=output_layer_function,
                                    bias=bias,
                                    init_type=init_type)
    
    def batchTrain(self):
        counter = 0
        while(counter <= self.data_train.shape[0]):
            upper_index = min(counter+self.batch_size, self.data_train.shape[0])
            batch_process = self.data_train[counter:upper_index]
            result = self.neural.forwardBatch(batch_process)

            
    def train_backprop(self) :
        for i in range(10):
            print("LEN")
            print((self.data_train[i, :]))
            self.neural.train(self.data_train[i, :], self.data_train_class[i], 1, 0.1)
    
    def saveANNasPickle(self, name):
        with open(f"./NeuralNetworks/{name}.pkl", "wb") as f:
            pickle.dump(self, f)
    
    def loadANNfromPickle(name):
        with open(f"./NeuralNetworks/{name}.pkl", "rb") as f:
            return pickle.load(f)