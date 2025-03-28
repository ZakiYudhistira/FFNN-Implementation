import numpy as np
    
def mean_squared_error(self, y_true: list, y_pred: list):
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    squared_errors = np.square(y_true - y_pred)
    mse = np.mean(squared_errors)
    return mse

def binary_cross_entropy(self, y_true: list, y_pred: list, epsilon=1e-15):
    # Clip predictions to avoid log(0) or log(1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def categorical_cross_entropy(self, y_true: list, y_pred: list, epsilon=1e-15):
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    cce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return cce