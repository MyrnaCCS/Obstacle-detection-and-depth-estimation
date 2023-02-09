import os
from collections import OrderedDict
import numpy as np
from Sequence import Sequence

class Dataset(object):
    def __init__(self, config):
        self.dataset_path = config.dataset_path
        self.data_train_val_sequences_path = config.data_train_val_path
        self.img_extension = config.img_extension
        self.train_val_sequences = OrderedDict() # Dictionary of Sequences
        
    def read_data(self):
        for sequence_path in self.data_train_val_sequences_path:
            self.train_val_sequences[sequence_path] = Sequence(os.path.join(self.dataset_path, sequence_path, 'rgb'),
                                                               ground_directory=os.path.join(self.dataset_path, sequence_path, 'ground'),
                                                               depth_directory=os.path.join(self.dataset_path, sequence_path, 'depth'), 
                                                               obstacles_directory=os.path.join(self.dataset_path, sequence_path, 'obstacles_3m'),
                                                               extension=self.img_extension)

    def generate_data(self, grd_depth_map_as_input=False):
        data = []
        for sequence_key in self.train_val_sequences:
            sequence = self.train_val_sequences[sequence_key]
            sequence_data = sequence.generate_sequence_data(grd_depth_map_as_input)
            data = np.append(data, sequence_data)
        return data