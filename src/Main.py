import DataLoader
import Engine
import pickle
import Misc as ms
from Classes import Configuration
import numpy as np

activation_functions_dict = {
    "relu": lambda x: np.maximum(0, x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "linear": lambda x: x,
    "softmax": lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True)
}

activation_functions_dict_derivative = {
    "relu": lambda x: np.where(x > 0, 1, 0),
    "sigmoid": lambda x: (lambda s: s * (1 - s))(1 / (1 + np.exp(-x))),
    "tanh": lambda x: 1 - np.tanh(x) ** 2,
    "linear": lambda x: np.ones_like(x),
    "softmax": lambda x: (lambda s: np.einsum('ij,ik->ijk', s, np.eye(s.shape[1])) - np.einsum('ij,ik->ijk', s, s))(
        np.exp(x - np.max(x, axis=-1, keepdims=True)) /
        np.sum(np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)
    )
}

loss_functions_dict = {
    "mean_squared_error": lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
    "binary_cross_entropy": lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10)),
    "categorical_cross_entropy": lambda y_true, y_pred: -np.mean(np.sum(y_true * np.log(y_pred + 1e-10), axis=1))
}

loss_functions_dict_derivative = {
    "mean_squared_error": lambda y_true, y_pred: 2 * (y_pred - y_true) / y_true.shape[0],
    "binary_cross_entropy": lambda y_true, y_pred: (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-10),
    "categorical_cross_entropy": lambda y_true, y_pred: -y_true / (y_pred + 1e-10) 
}

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
        main_engine = Engine.Engine.loadANNfromPickle(name_path)
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

def initiateEngine(config: Configuration, data_train, data_train_class):
    hidden_layer_activations = [activation_functions_dict[func] for func in config.hidden_layer_activations]
    hidden_layer_activations_derivative = [activation_functions_dict_derivative[func] for func in config.hidden_layer_activations]

    main_engine = Engine.Engine(config.hidden_layer_count,
                         config.hidden_layer_sizes,
                         hidden_layer_activations,
                         hidden_layer_activations_derivative,
                         activation_functions_dict[config.output_activation],
                         activation_functions_dict_derivative[config.output_activation],
                         config.bias,
                         config.init_type,
                         data_train,
                         data_train_class,
                         config.learning_rate,
                         config.epochs,
                         config.batch_size,
                         loss_functions_dict[config.loss_function],
                         loss_functions_dict_derivative[config.loss_function])
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
        for i in range(engine.epochs):
            print(f"Epoch {i+1}")
            engine.batchTrain()
        
main_config, flag = start_program()

data_train, data_train_class = loadPickle()
data_train_class = np.eye(np.max(data_train_class) + 1)[data_train_class]
print(data_train_class.shape)

if(flag):
    main_engine = main_config
else:
    main_engine = initiateEngine(main_config, data_train, data_train_class)
train(main_engine)