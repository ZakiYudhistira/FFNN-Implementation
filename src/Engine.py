from __future__ import annotations
from FuncDictionaries import activation_functions_dict, activation_functions_dict_derivative, loss_functions_dict, loss_functions_dict_derivative
from typing import Callable, List, Tuple
from graphviz import Digraph
from tqdm import tqdm
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split

# Savepath config
SAVE_PATH = "./NeuralNetworks/"
LOAD_PATH = "./NeuralNetworks/"
IMAGE_PATH = "./NeuralNetworks/Images/"

class Layer:
    def __init__(self,
                 neurons,
                 prev_neurons,
                 activation_function,
                 activation_function_derivative,
                 bias,
                 init_type="random-uniform"):
        self.n_neurons = neurons
        self.prev_n_nodes = prev_neurons+1
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        param1, param2, init_type, seed = init_type

        if init_type=="random-uniform":
            self.initiateWeightRDUniform(lower_bound=param1, upper_bound=param2, seed=seed)
        elif init_type=="random-normal":
            self.initiateWeightRDNormal(mean=param1, variance=param2, seed=seed)
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
    
    def updateWeights(self, input_array, learning_rate):
        self.delta = self.delta.astype(np.float64)
        input_array = input_array.astype(np.float64)
        gradient = input_array.T @ self.delta / input_array.shape[0]
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
    def __init__(self,
                 n_input: int = None,
                 n_output: int = None,
                 n_hiddenlayer: int = None,
                 hidden_layers_size: int = None,
                 hidden_layers_function = None,
                 hidden_layers_function_derivative = None,
                 output_layer_function = None,
                 output_layer_function_derivative = None,
                 bias: float = None,
                 init_type: tuple = None,
                 error_function = None,
                 error_function_derivative = None,
                 hidden_layers_function_strings: list[str] = None,
                 output_layer_function_string: list[str] = None,
                 error_function_string: list[str] = None,
                 neural_save: NeuralNetworkSave = None,
                 lambda_l1: float = 0.0,
                 lambda_l2: float = 0.0):
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        if neural_save is None:
            self.n_input = n_input
            self.n_output = n_output
            self.n_hiddenlayer = n_hiddenlayer
            self.hidden_layers_size = hidden_layers_size
            self.hidden_layers_function = hidden_layers_function
            self.hidden_layers_function_derivative = hidden_layers_function_derivative
            self.output_layer_function = output_layer_function
            self.output_layer_function_derivative = output_layer_function_derivative
            self.init_type = init_type
            self.error_function = error_function
            self.error_function_derivative = error_function_derivative
            self.hidden_layers_function_strings = hidden_layers_function_strings
            self.output_layer_function_string = output_layer_function_string
            self.error_function_string = error_function_string
            
            self.initiateLayers(bias)
        else:
            self.n_input = neural_save.n_input
            self.n_output = neural_save.n_output
            self.n_hiddenlayer = neural_save.n_hiddenlayer
            self.hidden_layers_size = neural_save.hidden_layers_size
            self.init_type = neural_save.init_type
            self.hidden_layers_function_strings = neural_save.hidden_layers_function_strings
            self.output_layer_function_string = neural_save.output_layer_function_string
            self.error_function_string = neural_save.error_function_string

            self.hidden_layers_function = [activation_functions_dict[func] for func in neural_save.hidden_layers_function_strings]
            self.hidden_layers_function_derivative = [activation_functions_dict_derivative[func] for func in neural_save.hidden_layers_function_strings]

            self.output_layer_function = activation_functions_dict[neural_save.output_layer_function_string]
            self.output_layer_function_derivative = activation_functions_dict_derivative[neural_save.output_layer_function_string]

            self.error_function = loss_functions_dict[neural_save.error_function_string]
            self.error_function_derivative = loss_functions_dict_derivative[neural_save.error_function_string]

            self.layers = neural_save.layer
            for i in range(len(self.hidden_layers_function_strings)):
                self.layers[i].activation_function = self.hidden_layers_function[i]
                self.layers[i].activation_function_derivative = self.hidden_layers_function_derivative[i]
            
            self.layers[-1].activation_function = self.output_layer_function
            self.layers[-1].activation_function_derivative = self.output_layer_function_derivative

    def initiateLayers(self, bias):
        self.layers = []
        self.layers.append(Layer(self.hidden_layers_size[0], self.n_input, self.hidden_layers_function[0], self.hidden_layers_function_derivative[0], bias, self.init_type))
        for i in range(1, self.n_hiddenlayer):
            self.layers.append(Layer(self.hidden_layers_size[i], self.hidden_layers_size[i-1], self.hidden_layers_function[i], self.hidden_layers_function_derivative[i], bias, self.init_type))
        self.layers.append(Layer(self.n_output, self.hidden_layers_size[-1], self.output_layer_function, self.output_layer_function_derivative, bias, self.init_type))
    
    def forward(self, input_matrix):
        for layer in self.layers:
            input_matrix = np.append(input_matrix, np.ones((input_matrix.shape[0], 1)), axis=1)
            input_matrix = layer.multiply(input_matrix)
        return input_matrix
    
    def backward(self, input_array, forwarded_result, expected_output, learning_rate):
        last_layer = self.layers[-1]
        # Calculate delta for the output layer
        last_layer.delta = (forwarded_result - expected_output) * self.output_layer_function_derivative(last_layer.output)

        # Backpropagate through the hidden layers
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            # Calculate the error term for the current layer
            error_term = np.dot(next_layer.delta, next_layer.weight_matrix[:-1, :].T)
            layer.delta = error_term * layer.activation_function_derivative(layer.output)

        batch_size = input_array.shape[0]
        for i, layer in enumerate(self.layers):
            if i == 0:
                input_with_bias = np.hstack((input_array, np.ones((batch_size, 1))))
            else:
                prev_output = self.layers[i - 1].output
                input_with_bias = np.hstack((prev_output, np.ones((batch_size, 1))))

            layer.updateWeights(input_with_bias, learning_rate)

    def train(self, inputs_batch, expected_outputs, learning_rate):
        forward_result = self.forward(inputs_batch)
        self.backward(inputs_batch, forward_result, expected_outputs, learning_rate)
    
    def visualizeNetwork(self, filename='neural_network'):
        dot = Digraph(comment='Neural Network Visualization')
        dot.attr(rankdir='LR')
        dot.attr('graph', nodesep='1.5') 
        dot.attr('graph', ranksep='10')
        dot.attr('graph', ordering='in')
        
        # input layers
        with dot.subgraph(name='cluster_0') as c:
            c.attr(label='Input Layer')
            c.attr(rank='same') 
            for i in range(self.n_input):
                c.node(f'i{i}', f'Input {i}')
            c.node(f'i{self.n_input}', 'Bias')

        
        # hidden layers
        for layer_idx, layer in enumerate(self.layers[:-1], 1):
            with dot.subgraph(name=f'cluster_{layer_idx}') as c:
                c.attr(label=f'Hidden Layer {layer_idx}')
                c.attr(rank='same')  # Force same rank
                if layer.delta is None:
                    layer.delta = np.random.rand(layer.n_neurons) # Debugging purposes
                    print("Layer delta is None, assigning random values.")
                for neuron_idx in range(layer.n_neurons):
                    delta_val = f'\nδ={layer.delta[neuron_idx]:.4f}' if layer.delta is not None else ''
                    func_name = self.hidden_layers_function_strings[layer_idx-1]
                    c.node(f'h{layer_idx}_{neuron_idx}', 
                        f'Neuron {neuron_idx}\n{func_name}{delta_val}')
                c.node(f'h{layer_idx}_bias', 'Bias')

        
        # output layer
        with dot.subgraph(name=f'cluster_{len(self.layers)}') as c:
            c.attr(label='Output Layer')
            c.attr(rank='same')  # Force same rank
            output_layer = self.layers[-1]
            if output_layer.delta is None:
                print("Output layer delta is None, assigning random values.")
                output_layer.delta = np.random.rand(output_layer.n_neurons)
            for neuron_idx in range(output_layer.n_neurons):
                delta_val = f'\nδ={output_layer.delta[neuron_idx]:.4f}' if output_layer.delta is not None else ''
                func_name = self.output_layer_function_string
                c.node(f'o{neuron_idx}', 
                    f'Output {neuron_idx}\n{func_name}{delta_val}')
        
        # Add edges with weights
        # Connect input to first hidden layer
        for i in range(self.n_input + 1):
            for j in range(self.layers[0].n_neurons):
                weight = self.layers[0].weight_matrix[i, j]
                dot.edge(f'i{i}', f'h1_{j}', f'{weight:.2f}')
        
        # hidden layers weight
        for layer_idx in range(1, len(self.layers)-1):
            prev_layer = self.layers[layer_idx-1]
            curr_layer = self.layers[layer_idx]
            for i in range(prev_layer.n_neurons + 1):
                node_id = f'h{layer_idx}_{i}'
                if i == prev_layer.n_neurons:
                    node_id = f'h{layer_idx}_bias'
                for j in range(curr_layer.n_neurons):
                    weight = curr_layer.weight_matrix[i, j]
                    dot.edge(node_id, f'h{layer_idx+1}_{j}', f'{weight:.2f}')
        
        # output layers weight
        last_hidden_idx = len(self.layers) - 1
        last_hidden = self.layers[-2]
        output_layer = self.layers[-1]
        for i in range(last_hidden.n_neurons + 1):
            node_id = f'h{last_hidden_idx}_{i}'
            if i == last_hidden.n_neurons:
                node_id = f'h{last_hidden_idx}_bias'
            for j in range(output_layer.n_neurons):
                weight = output_layer.weight_matrix[i, j]
                dot.edge(node_id, f'o{j}', f'{weight:.2f}')
        
        # Saving
        output_path = dot.render(IMAGE_PATH+filename, view=True, format='svg')
        output_path = output_path.replace('.svg', '')
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def displayWeightDistribution(self, layer_idx: list[int]):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(layer_idx), 1, figsize=(8, 5 * len(layer_idx)))
        if len(layer_idx) == 1:
            axes = [axes]
        for ax, number in zip(axes, layer_idx):
            layer = self.layers[number]
            weights = layer.weight_matrix.flatten()
            ax.hist(weights, bins=30, alpha=0.75, color='blue', edgecolor='black')
            if number == len(self.layers)-1:
                ax.set_title(f'Weight Distribution for Output Layer')
            else:
                ax.set_title(f'Weight Distribution for Hidden Layer {number+1}')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.grid(True)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def displayDeltaDistribution(self, layer_idx: list[int]):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(layer_idx), 1, figsize=(8, 5 * len(layer_idx)))
        if len(layer_idx) == 1:
            axes = [axes]
        
        has_data = False
        
        for ax, number in zip(axes, layer_idx):
            layer = self.layers[number]
            if layer.delta is None:
                print(f"Layer {number} has no delta values to display.")
                continue
                
            has_data = True
            deltas = layer.delta.flatten()
            ax.hist(deltas, bins=30, alpha=0.75, color='blue', edgecolor='black')
            if number == len(self.layers)-1:
                ax.set_title(f'Delta Distribution for Output Layer')
            else:
                ax.set_title(f'Delta Distribution for Hidden Layer {number+1}')
            ax.set_xlabel('Delta Value')
            ax.set_ylabel('Frequency')
            ax.grid(True)
        
        if not has_data:
            print("No delta values to display for the selected layers.")
            plt.close(fig)
        else:
            plt.subplots_adjust(hspace=0.5)
            plt.show()

    def printSpec(self):
        print(f"Input: {self.n_input}")
        print(f"Output: {self.n_output}")
        print(f"Hidden Layer Count: {self.n_hiddenlayer}")
        print(f"Hidden Layer Sizes: {self.hidden_layers_size}")
        print(f"Hidden Layer Functions: {self.hidden_layers_function_strings}")
        print(f"Output Layer Function: {self.output_layer_function_string}")
        print(f"Error Function: {self.error_function_string}")
                
class NeuralNetworkSave:
    def __init__(self, neural:NeuralNetwork):
        self.n_input = neural.n_input
        self.n_output = neural.n_output
        self.n_hiddenlayer = neural.n_hiddenlayer
        self.hidden_layers_size = neural.hidden_layers_size
        self.init_type = neural.init_type
        self.hidden_layers_function_strings = neural.hidden_layers_function_strings
        self.output_layer_function_string = neural.output_layer_function_string
        self.error_function_string = neural.error_function_string

        self.layer = neural.layers

        for layer in self.layer:
            layer.activation_function = None
            layer.activation_function_derivative = None

class Engine():
    def __init__(self,
                 data_train,
                 data_train_class,
                 learning_rate : float,
                 epochs : int,
                 batch_size : int,
                 neural_network : NeuralNetwork,
                 error_function: Callable[[np.ndarray, np.ndarray], float]):
        self.data_train = data_train
        self.data_train_class = data_train_class
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.error_function = error_function

        self.neural = neural_network
    
    def batchTrain(self, debug=False):
        # Split the data into training and validation sets (80% train, 20% validation)
        self.data_train, self.data_val, self.data_train_class, self.data_val_class = train_test_split(
            self.data_train, self.data_train_class, test_size=0.2, random_state=42
        )
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        for i in range(self.epochs):
            print(f"Epoch {i+1}/{self.epochs}")
            batch_count = (self.data_train.shape[0] + self.batch_size - 1) // self.batch_size  # Calculate total batches
            
            if debug:
                print("Debug mode: Progress bar disabled.")
                for batch_counter in range(1, batch_count + 1):
                    start_index = (batch_counter - 1) * self.batch_size
                    upper_index = min(start_index + self.batch_size, self.data_train.shape[0])
                    batch_data = self.data_train[start_index:upper_index]
                    expected_result = self.data_train_class[start_index:upper_index]

                    result = self.neural.train(batch_data, expected_result, self.learning_rate)
                    print(f"Batch {batch_counter}/{batch_count} processed.")
            else:
                with tqdm(total=batch_count, desc="Training Progress", unit="batch") as pbar:
                    for batch_counter in range(1, batch_count + 1):
                        start_index = (batch_counter - 1) * self.batch_size
                        upper_index = min(start_index + self.batch_size, self.data_train.shape[0])
                        batch_data = self.data_train[start_index:upper_index]
                        expected_result = self.data_train_class[start_index:upper_index]

                        self.neural.train(batch_data, expected_result, self.learning_rate)
                        pbar.update(1)

            val_predictions = self.neural.forward(self.data_val)
            val_loss = self.error_function(self.data_val_class, val_predictions)
            print(f"Validation Loss after Epoch {i+1}: {val_loss}")
        print("Finished training")
        print("Final Validation Loss: ", val_loss)
        print("Final Validation Predictions: ", val_predictions)
            
    def train_backprop(self) :
        for i in range(10):
            print((self.data_train[i, :]))
            self.neural.train(self.data_train[i, :], self.data_train_class[i], 1, 0.1)
    
    def saveANNtoPickle(self, name):
        neural_save = NeuralNetworkSave(self.neural)
        with open(f"{SAVE_PATH}{name}", "wb") as f:
            pickle.dump(neural_save, f)
    
    def loadANNfromPickle(name):
        with open(f"{LOAD_PATH}{name}", "rb") as f:
            neural_save_config = pickle.load(f)
            neural = NeuralNetwork(neural_save=neural_save_config)
            return neural
    
    def visualizeNetwork(self, filename='neural_network'):
        self.neural.visualizeNetwork(filename)