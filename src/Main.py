import DataLoader
import Engine
import pickle
import Misc as ms
from Classes import Configuration
import numpy as np

def savePickle():
    loader = DataLoader.DataLoader('./data/mnist_784.arff')
    data_train, data_train_class = loader.load_data()
    data_train_temp = open('./data/test_data_train', 'wb')
    pickle.dump(data_train, data_train_temp)
    data_train_temp.close()

    data_train_class_temp = open('./data/test_data_train_class', 'wb')
    pickle.dump(data_train_class, data_train_class_temp)
    data_train_class_temp.close()

def loadPickle():
    data_train_temp = open('./data/test_data_train', 'rb')
    data_train = pickle.load(data_train_temp)
    data_train_temp.close()

    data_train_class_temp = open('./data/test_data_train_class', 'rb')
    data_train_class = pickle.load(data_train_class_temp)
    data_train_class_temp.close()

    return data_train, data_train_class

def start_program():
    print(">>> Welcome to Neural Network <<<")
    print(">>> Do you want to load your custom configuration?")
    print(">>> 1. Yes")
    print(">>> 2. No")
    print(">>> Choose: ", end="")
    while True:
        try:
            choice = int(input())
            if (choice == 1 or choice == 2):
                break
        except:
            print(">>> Invalid input")
    if(choice == 1):
        file_name = input(">>> Input your configuration file: ")
        config = Configuration.loadConfigfromJSON(f"./config/{file_name}")
    else:
        print(">>> Using custom configuration")
        config_name = input(">>> Enter config name: ")
        batch_size = ms.getPositiveInteger(">>> Input batch size: ")
        learning_rate = ms.getPositiveFLoat(">>> Input learning rate: ")
        epochs = ms.getPositiveInteger(">>> Input epochs: ")
        loss_function = ms.getLossFunction()
        hidden_layer_count = ms.getPositiveInteger(">>> Input hidden layer count: ")
        print(">>> Input hidden layer sizes: ")
        hidden_layer_sizes = []
        for i in range(hidden_layer_count):
            hidden_layer_sizes.append(ms.getPositiveInteger(f">>> Input hidden layer {i+1} size: "))
        hidden_layer_activations = []
        print(">>> Input hidden layer activations: ")
        for i in range(hidden_layer_count):
            print(f">>> Input hidden layer {i+1} activation: ")
            hidden_layer_activations.append(ms.getHiddenFunction())
        print(">>> Input output activation: ")
        output_activation = ms.getHiddenFunction()
        bias = ms.getPositiveFLoat(">>> Input bias: ")
        init_type_information = ms.getInitType()

        config = Configuration(config_name=config_name,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            loss_function=loss_function,
                            hidden_layer_count=hidden_layer_count,
                            hidden_layer_sizes=hidden_layer_sizes,
                            hidden_layer_activations=hidden_layer_activations,
                            output_activation=output_activation,
                            bias=bias,
                            init_type=init_type_information,
                            data_train=None,
                            data_train_class=None)
    
    print(">>> Configuration <<<")
    print("Using configuration:", config.config_name)
    print("Batch size: ", config.batch_size)
    print("Learning rate: ", config.learning_rate)
    print("Epochs: ", config.epochs)
    print("Loss function: ", config.loss_function)
    print("Hidden layer count: ", config.hidden_layer_count)
    print("Hidden layer sizes: ", config.hidden_layer_sizes)
    print("Hidden layer activations: ", config.hidden_layer_activations)
    print("Output activation: ", config.output_activation)
    print("Bias: ", config.bias)
    print("Init type: ", config.init_type)

    save_flag = input(">>> Do you want to save this configuration? (Y/N): ")

    if(save_flag.upper() == "Y"):
        Configuration.saveConfigtoJSON(config, f"./config/{config_name}.json")

    return config
        
start_program()

data_train, data_train_class = loadPickle()

array_input = np.array([2,2,3,4,5,-1])
outputs = 7
hidden_layers = 5
hidden_sizes = [4, 3, 1, 6, 4]
bias=5
init_type="he"

function = lambda x:x
hidden_function = [function, function, function, function, function]

engine = Engine.Engine(n_hiddenlayer=hidden_layers,
                       hidden_layers_size=hidden_sizes,
                       hidden_layers_function=hidden_function,
                       output_layer_function=function,
                       bias=bias,
                       init_type=init_type,
                       data_train=data_train,
                       data_train_class=data_train_class,
                       learning_rate=0.01)
# engine.batch_train()
engine.train_backprop()