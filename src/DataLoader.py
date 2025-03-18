import pandas as pd
import time
from scipy.io import arff

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        print("loading dataset from", self.path)
        start_time = time.time()

        # load the data
        data_arff = arff.loadarff(self.path)
        data = pd.DataFrame(data_arff[0]).to_numpy()

        # separate the class column
        instance_class = data[:,-1]
        instance_class = instance_class.astype(int)
        data = data[:,:-1]

        end_time = time.time()
        print("data loaded in", "{:.2f}".format(end_time-start_time), "seconds")
        return data, instance_class