import json
from classes import *

def load_init(name : str) :    
    # Open and read a JSON file
    path = "data/"
    with open(path+name, "r") as file:
        data = json.load(file)  # Parse JSON into a Python dictionary

    configuration = Configuration(data["training_parameters"]["batch_size"], data["training_parameters"]["learning_rate"], data["training_parameters"]["epochs"], data["training_parameters"]["loss_function"])

    list_of_layers = []
    for i in range (data["architecture"]["layers"]) :
        if (i == 0) :
            activation_functions = "linear"
        else :
            activation_functions = data["architecture"]["activation_functions"][i-1]
            
        for j in range (len(data["weights"])) :
            if (data["weights"][j]["layer_index"] == i) :
                type = data["weights"][j]["type"]
                if (type == "zero") :
                    param = [0,0,0]
                else :
                    param = data["weights"][j]["parameter"]
                break
        
        neuron_length = data["architecture"]["neurons"][i]
        list_of_layers.append(Layer(neuron_length, type, param, activation_functions))
            
        for j in range (neuron_length) :
            list_of_layers[i].neurons[j].value = data["neurons"][i][j]
            
    
    # Print the loaded data
    return list_of_layers, configuration

load_init("input.json")