import DataLoader
import Engine
import pickle
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

data_train, data_train_class = loadPickle()

# instance = 784

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