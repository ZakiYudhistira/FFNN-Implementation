import DataLoader
import Engine
import pickle
import Misc as ms
from Classes import Configuration
import numpy as np
from FuncDictionaries import activation_functions_dict, activation_functions_dict_derivative, loss_functions_dict, loss_functions_dict_derivative

# Load config
DATASET_LOAD_PATH = './data/mnist_784.arff'
PICKLED_TRAIN_DATA_SAVE_PATH = './data/test_data_train'
PICKLED_CLASS_DATA_SAVE_PATH = './data/test_data_train_class'

def savePickle():
    loader = DataLoader.DataLoader(DATASET_LOAD_PATH)
    data_train, data_train_class = loader.load_data()
    data_train_temp = open(PICKLED_TRAIN_DATA_SAVE_PATH, 'wb')
    pickle.dump(data_train, data_train_temp)
    data_train_temp.close()

    data_train_class_temp = open(PICKLED_CLASS_DATA_SAVE_PATH, 'wb')
    pickle.dump(data_train_class, data_train_class_temp)
    data_train_class_temp.close()

def loadPickle():
    data_train_temp = open(PICKLED_TRAIN_DATA_SAVE_PATH, 'rb')
    data_train = pickle.load(data_train_temp)
    data_train_temp.close()

    data_train_class_temp = open(PICKLED_CLASS_DATA_SAVE_PATH, 'rb')
    data_train_class = pickle.load(data_train_class_temp)
    data_train_class_temp.close()

    return data_train, data_train_class

def start_program():
    print(">>> Welcome to Neural Network <<<")
    print(">>> Please choose an action")
    print(">>> 1. Load a Neural Network Configuration")
    print(">>> 2. Load a Neural Network Spesification JSON")
    print(">>> 3. Create a new Neural Network Configuration")
    print(">>> Choose: ", end="")
    while True:
        try:
            choice = int(input())
            if (choice == 1 or choice == 2 or choice == 3):
                break
        except:
            print(">>> Invalid input")
    if(choice == 2):
        file_name = input(">>> Input your configuration file: ")
        config = Configuration.loadConfigfromJSON(f"./config/{file_name}")
    elif(choice == 3):
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
        
        save_flag = input(">>> Do you want to save this configuration? (Y/N): ")

        if(save_flag.upper() == "Y"):
            Configuration.saveConfigtoJSON(config, f"./config/{config_name}.json")
    elif(choice == 1):
        name_path = input(">>> Input your Artificial Neural Network Configuration file: ")
        print(">>> Load configuration from pickle <<<")
        neural = Engine.Engine.loadANNfromPickle(name_path)
        main_engine = initiateEngine(config=None, data_train=None, data_train_class=None, neural=neural)
        return main_engine, True


    print(">>> Configuration <<<")
    print("Using configuration:", config.config_name)
    print("Batch size:", config.batch_size)
    print("Learning rate:", config.learning_rate)
    print("Epochs:", config.epochs)
    print("Loss function:", config.loss_function)
    print("Hidden layer sizes:", config.hidden_layer_sizes)
    print("Hidden layer count:", config.hidden_layer_count)
    print("Hidden layer activations:", config.hidden_layer_activations)
    print("Output activation:", config.output_activation)
    print("Bias:", config.bias)
    print("Init type:", config.init_type)

    return config, False

def initiateEngine(config: Configuration, data_train, data_train_class, neural: Engine.NeuralNetwork = None):
    if neural is None:
        hidden_layer_activations = [activation_functions_dict[func] for func in config.hidden_layer_activations]
        hidden_layer_activations_derivative = [activation_functions_dict_derivative[func] for func in config.hidden_layer_activations]

        neural = Engine.NeuralNetwork(n_input=data_train.shape[1],
                                    n_output=data_train_class.shape[1],
                                    n_hiddenlayer=config.hidden_layer_count,
                                    hidden_layers_size=config.hidden_layer_sizes,
                                    hidden_layers_function=hidden_layer_activations,
                                    hidden_layers_function_derivative=hidden_layer_activations_derivative,
                                    output_layer_function=activation_functions_dict[config.output_activation],
                                    output_layer_function_derivative=activation_functions_dict_derivative[config.output_activation],
                                    bias=config.bias,
                                    init_type=config.init_type,
                                    error_function=loss_functions_dict[config.loss_function],
                                    error_function_derivative=loss_functions_dict_derivative[config.loss_function],
                                    hidden_layers_function_strings=config.hidden_layer_activations,
                                    output_layer_function_string=config.output_activation,
                                    error_function_string=config.loss_function)
                                    

        main_engine = Engine.Engine(data_train=data_train,
                                data_train_class=data_train_class,
                                learning_rate=config.learning_rate,
                                epochs=config.epochs,
                                batch_size=config.batch_size,
                                neural_network=neural,
                                error_function=loss_functions_dict[config.loss_function])
    else:
        if(not True):
            ms.getPositiveFLoat(">>> Input learning rate: ")
            ms.getPositiveInteger(">>> Input epochs: ")
            ms.getPositiveInteger(">>> Input batch size: ")
        else:
            learning_rate = 0.01 # ms.getPositiveFLoat(">>> Input learning rate: ")
            epochs = 10 # ms.getPositiveInteger(">>> Input epochs: ")
            batch_size = 32 # ms.getPositiveInteger(">>> Input batch size: ")
        main_engine = Engine.Engine(data_train=data_train,
                                data_train_class=data_train_class,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                batch_size=batch_size,
                                neural_network=neural,
                                error_function=neural.error_function)
    return main_engine

def train(engine):
    print(">>> Training <<<")
    print("Do you want to train the ANN? (Y/N): ", end="")
    while True:
        try:
            choice = input()
            if (choice.upper() == "Y" or choice.upper() == "N"):
                break
        except:
            print(">>> Invalid input")
    if(choice.upper() == "Y"):
        print(">>> Begin training <<<")
        print(f">>> Batch size: {engine.batch_size}")
        engine.batchTrain()
    else:
        print(">>> Skip training <<<")
    print(">>> Do you want to save the ANN? (Y/N): ", end="")
    while True:
        try:
            choice = input()
            if (choice.upper() == "Y" or choice.upper() == "N"):
                break
        except:
            print(">>> Invalid input")
    if(choice.upper() == "Y"):
        name = input(">>> Enter name for the ANN: ")
        engine.saveANNtoPickle(name)
        print(f">>> {name} ANN saved <<<")
    else:
        print(">>> ANN not saved <<<")

def visualize(engine):
    print(">>> Visualizing <<<")
    print("Do you want to visualize the ANN? (Y/N): ", end="")
    while True:
        try:
            choice = input()
            if (choice.upper() == "Y" or choice.upper() == "N"):
                break
        except:
            print(">>> Invalid input")
    if(choice.upper() == "Y"):
        file_name = input(">>> Enter name for the File: ")
        engine.visualizeNetwork(file_name)
    else:
        print(">>> Skip visualization <<<")

def showWeightDistribution(engine):
    print(">>> Show weight distribution <<<")
    print("Do you want to show weight distribution? (Y/N): ", end="")
    while True:
        try:
            choice = input()
            if (choice.upper() == "Y" or choice.upper() == "N"):
                break
        except:
            print(">>> Invalid input")
    if(choice.upper() == "Y"):
        print(f">>> There are {engine.neural.n_hiddenlayer} hidden layers in the ANN")
        layers_to_show = input(">>> Enter layers to show (eg:0 1 2 3 4 5): ")
        layers_to_show = layers_to_show.split()
        layers_to_show = [int(layer) for layer in layers_to_show]
        engine.neural.displayWeightDistribution(layers_to_show)
    else:
        print(">>> Skip weight distribution <<<")

    print(">>> Show delta distribution <<<")
    print("Do you want to show delta distribution? (Y/N): ", end="")
    while True:
        try:
            choice = input()
            if (choice.upper() == "Y" or choice.upper() == "N"):
                break
        except:
            print(">>> Invalid input")
    if(choice.upper() == "Y"):
        print(f">>> There are {engine.neural.n_hiddenlayer} hidden layers in the ANN")
        layers_to_show = input(">>> Enter layers to show (eg:0 1 2 3 4 5): ")
        layers_to_show = layers_to_show.split()
        layers_to_show = [int(layer) for layer in layers_to_show]
        engine.neural.displayDeltaDistribution(layers_to_show)
        

data_train, data_train_class = loadPickle()
data_train_class = np.eye(np.max(data_train_class) + 1)[data_train_class]

# main_config, flag = start_program()
# if(flag):
#     main_engine = main_config
#     main_engine.data_train = data_train
#     main_engine.data_train_class = data_train_class
# else:
#     main_engine = initiateEngine(main_config, data_train, data_train_class)
# train(main_engine)
# showWeightDistribution(main_engine)

# neural = Engine.Engine.loadANNfromPickle("New2")
# main_engine = initiateEngine(config=None, data_train=data_train, data_train_class=data_train_class, neural=neural)
# main_engine.neural.displayWeightDistribution([0, 1, 2, 3, 4])
# main_engine.batchTrain()
# main_engine.neural.displayWeightDistribution([0, 1, 2, 3, 4])


# main_engine = initiateEngine(Configuration.loadConfigfromJSON(f"./config/config1_length.json"), data_train, data_train_class)
# main_engine.batchTrain()
# main_engine.neural.displayWeightDistribution([0, 1, 2, 8])
# main_engine.neural.displayDeltaDistribution([0, 1, 2, 8])
# main_engine.saveANNtoPickle("config1_length")

# main_engine = initiateEngine(Configuration.loadConfigfromJSON(f"./config/config2_length.json"), data_train, data_train_class)
# main_engine.batchTrain()
# main_engine.saveANNtoPickle("config2_length")
# main_engine.neural.displayWeightDistribution([0, 1, 2, 3])
# main_engine.neural.displayDeltaDistribution([0, 1, 2, 3])

# main_engine = initiateEngine(Configuration.loadConfigfromJSON(f"./config/config3_length.json"), data_train, data_train_class)
# # main_engine.batchTrain()
# neural = Engine.Engine.loadANNfromPickle("config3_length")
# neural.displayWeightDistribution([0, 1, 2])
# neural.displayDeltaDistribution([0, 1, 2])

main_engine = initiateEngine(Configuration.loadConfigfromJSON(f"./config/config4_length.json"), data_train, data_train_class)
main_engine.batchTrain()
main_engine.saveANNtoPickle("config4_length")
main_engine.neural.displayWeightDistribution([0, 1, 2, 3, 4])
main_engine.neural.displayDeltaDistribution([0, 1, 2, 3, 4])