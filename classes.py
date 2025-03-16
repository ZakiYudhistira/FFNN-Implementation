import numpy as np

# Dictionary of Activation Functions
activation_functions = {
    "relu": lambda x: np.maximum(0, x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "linear": lambda x: x,
    "softmax": lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True)
}

class Configuration:
    def __init__(self, batch_size:int, learning_rate:float, epochs:int, loss_function:str):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs 
        self.loss_function = loss_function

class Neuron:
    def __init__(self, value=0):
        self.value = value

class Weight:
    def __init__(self, weight_type: str, parameter: list[float, float, float]):
        self.type = weight_type
        self.parameter = parameter

class Layer:
    def __init__(self, num_neurons: int, weight_type: str, weight_param: list[float, float, float], activation_name: str):
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.weight = Weight(weight_type, weight_param)
        
        # Assign activation function dynamically
        self.activation = activation_functions.get(activation_name, None)
        
        if self.activation is None:
            raise ValueError(f"Activation function '{activation_name}' is not defined.")

    def apply_activation(self, inputs):
        """Applies the selected activation function to the inputs."""
        return self.activation(inputs)

# Example Usage
# layer = Layer(num_neurons=4, weight_type="uniform", weight_param=(0, 1, 2), activation_name="relu")
# inputs = np.array([-2, 0, 3])
# output = layer.apply_activation(inputs)

# print(f"Layer Activation Output: {output}")  # Expected: [0, 0, 3] for ReLU

