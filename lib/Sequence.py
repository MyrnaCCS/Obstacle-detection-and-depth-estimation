import os
from glob import glob
from itertools import tee, islice
from Element import Element

class Sequence(object):
    def __init__(self, input_directory, ground_directory, depth_directory, obstacles_directory, extension):
        self.input_paths = sorted(glob(os.path.join(input_directory, '*' + '.' + extension)))
        self.grd_paths = sorted(glob(os.path.join(ground_directory, '*' + '.' + extension)))
        self.depth_paths = sorted(glob(os.path.join(depth_directory, '*' + '.' + extension)))
        self.obstacles_paths = sorted(glob(os.path.join(obstacles_directory, '*' + '.' + 'txt')))

    @staticmethod
    def nwise(iterable, n=2):
        iters = (islice(each, i, None) for i, each in enumerate(tee(iterable, n)))
        return zip(*iters)
    
    def generate_sequence_data(self, grd_depth_map_as_input=False):
        data = []
        for input_set, grd_set, depth_set, obs_set in zip(self.nwise(self.input_paths, 1), self.nwise(self.grd_paths, 1), self.nwise(self.depth_paths, 1), self.nwise(self.obstacles_paths, 1)):
            feature_set = [input_set, grd_set]
            label_set = [depth_set, obs_set]
            data.append(Element(feature_set, label_set, grd_depth_map_as_input))
        return data