import numpy as np

class InitializationMethode:
    def __init__(self):
        pass
    
    def xavier_initialization(n_input, n_output) -> list:
        """
        - n_input: banyak neuron input
        - n_ouput: banyak neuron output
        """
        limit = np.sqrt(6 / (n_input + n_output))
        weights = np.random.uniform(-limit, limit, size=(n_input, n_output))
        return weights
    
    def he_initialization(n_input: int, n_output: int) -> list:
        """
        - n_input: banyak neuron input
        - n_ouput: banyak neuron output
        """
        std_dev = np.sqrt(2 / n_input)
        weights = np.random.normal(0, std_dev, size=(n_input, n_output))
        return weights

class Normalization:
    def __init__(self):
        pass
    
    def RMSnorm(self, dim, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(dim)

        def forward(self, x):
            rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
            
            normalized = x / rms
            return normalized * self.gamma
        
        return forward()