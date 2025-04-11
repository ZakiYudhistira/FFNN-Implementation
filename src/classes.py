import numpy as np
import json

class Configuration:
    def __init__(self,
                 config_name:str,
                 batch_size:int,
                 learning_rate:float,
                 epochs:int,
                 loss_function:str,
                 hidden_layer_count:int,
                 hidden_layer_sizes:list[int],
                 hidden_layer_activations:list[str],
                 output_activation:str,
                 bias:float,
                 init_type:tuple,
                 data_train:np.ndarray,
                 data_train_class:np.ndarray):
        self.config_name = config_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs 
        self.loss_function = loss_function
        self.hidden_layer_count = hidden_layer_count
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activations = hidden_layer_activations
        self.output_activation = output_activation
        self.bias = bias
        self.init_type = init_type
        self.data_train = data_train
        self.data_train_class = data_train_class
        
    def to_dict(self):
        return {
            "config_name": self.config_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "loss_function": self.loss_function,
            "hidden_layer_count": self.hidden_layer_count,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "hidden_layer_activations": [func for func in self.hidden_layer_activations],
            "output_activation": self.output_activation,
            "bias": self.bias,
            "init_type": self.init_type,
            "data_train": self.data_train.tolist() if isinstance(self.data_train, np.ndarray) else self.data_train,
            "data_train_class": self.data_train_class.tolist() if isinstance(self.data_train_class, np.ndarray) else self.data_train_class
        }
    
    def saveConfigtoJSON(config, filepath):
        config_dict = config.to_dict()

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=4)

        print(f"Configuration saved to {filepath}")
    
    def loadConfigfromJSON(filepath):
        with open(filepath, "r") as f:
            config_dict = json.load(f)
            return Configuration(config_name=config_dict["config_name"],
                             batch_size=config_dict["batch_size"],
                             learning_rate=config_dict["learning_rate"],
                             epochs=config_dict["epochs"],
                             loss_function=config_dict["loss_function"],
                             hidden_layer_count=config_dict["hidden_layer_count"],
                             hidden_layer_sizes=config_dict["hidden_layer_sizes"],
                             hidden_layer_activations=config_dict["hidden_layer_activations"],
                             output_activation=config_dict["output_activation"],
                             bias=config_dict["bias"],
                             init_type=config_dict["init_type"],
                             data_train=None,
                             data_train_class=None) 


        
        return Configuration(config_name=config_dict["config_name"],
                             batch_size=config_dict["batch_size"],
                             learning_rate=config_dict["learning_rate"],
                             epochs=config_dict["epochs"],
                             loss_function=config_dict["loss_function"],
                             hidden_layer_count=config_dict["hidden_layer_count"],
                             hidden_layer_sizes=config_dict["hidden_layer_sizes"],
                             hidden_layer_activations=config_dict["hidden_layer_activations"],
                             output_activation=config_dict["output_activation"],
                             bias=config_dict["bias"],
                             init_type=config_dict["init_type"],
                             data_train=np.array(config_dict["data_train"]),
                             data_train_class=np.array(config_dict["data_train_class"]))