import os
from glob import glob
from itertools import tee, islice
from Element import Element

class Sequence(object):
    def __init__(self, input_directory, depth_directory, obstacles_directory, extension):
        self.input_paths = sorted(glob(os.path.join(input_directory, '*' + '.' + extension)))
        self.depth_paths = sorted(glob(os.path.join(depth_directory, '*' + '.' + extension)))
        self.obstacles_paths = sorted(glob(os.path.join(obstacles_directory, '*' + '.' + 'txt')))

    @staticmethod
    def nwise(iterable, n=2):
        iters = (islice(each, i, None) for i, each in enumerate(tee(iterable, n)))
        return zip(*iters)
    
    def generate_sequence_data(self):
        data = []
        for input_set, depth_set, obs_set in zip(self.nwise(self.input_paths, 1), self.nwise(self.depth_paths, 1), self.nwise(self.obstacles_paths, 1)):
            label_set = [depth_set, obs_set]
            data.append(Element(input_set, label_set))
        return data