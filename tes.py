import numpy as np

func = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid function

# Correct usage of np.ndarray: Define shape, then populate values
tes = np.ndarray(shape=(6,), dtype=float)  # Create an uninitialized ndarray with 6 elements
tes[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # Assign values to it

print(func(tes))
