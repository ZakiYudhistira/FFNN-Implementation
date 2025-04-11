import numpy as np

activation_functions_dict = {
    "relu": lambda x: np.maximum(0, x),
    "sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
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