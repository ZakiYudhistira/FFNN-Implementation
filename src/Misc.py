def getPositiveInteger(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            else:
                print("Value must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

def getPositiveFLoat(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            else:
                print("Value must be a positive number.")
        except ValueError:
            print("Invalid input. Please enter a positive number.")

def getLossFunction():
    while True:
        print("1. mean_squared_error\n2. binary_cross_entropy\n3. categorical_cross_entropy")
        loss_function = str(input(">>> Input loss function: "))
        loss_dict = {
            "1": "mean_squared_error",
            "2": "binary_cross_entropy",
            "3": "categorical_cross_entropy"
        }
        if loss_function in loss_dict.keys():
            return loss_dict[loss_function]
        else:
            print("Invalid loss function. Please choose from 'mse' or 'crossentropy'.")

def getHiddenFunction():
    print("1. relu\n2. sigmoid\n3. tanh\n4. linear\n5. softmax")
    hidden_function = str(input(">>> Input layer activation: "))
    hidden_dict = {
        "1": "relu",
        "2": "sigmoid",
        "3": "tanh",
        "4": "linear",
        "5": "softmax"
    }
    return hidden_dict[hidden_function]

def getInitType():
    print("1. random-uniform\n2. random-normal\n3. he\n4. xavier\n5. zeros")
    init_type = str(input(">>> Input init type: "))
    init_dict = {
        "1": "random-uniform",
        "2": "random-normal",
        "3": "he",
        "4": "xavier",
        "5": "zero"
    }
    if(init_type == "1"):
        while(True):
            lower_bound = float(input(">>> Input lower bound: "))
            upper_bound = float(input(">>> Input upper bound: "))
            seed = int(input(">>> Input seed: "))
            if(lower_bound > upper_bound):
                print(">>> Lower bound must be less than upper bound")
            else:
                break
        return lower_bound, upper_bound, init_dict[init_type], seed
    elif(init_type == "2"):
        mean = float(input(">>> Input mean: "))
        variance = float(input(">>> Input variance: "))
        seed = int(input(">>> Input seed: "))
        return mean, variance, init_dict[init_type], seed
    elif(init_type == "3" or init_type == "4" or init_type == "5"):
        return None, None, init_dict[init_type], 0