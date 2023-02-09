import cv2 as cv
import numpy as np

def select_anchor_index(width, height):
    anchor1 = [55, 125]
    anchor2 = [218, 154]
    iou1 = min(width, anchor1[0]) * min(height, anchor1[1]) / (width * height + anchor1[0] * anchor1[1])
    iou2 = min(width, anchor2[0]) * min(height, anchor2[1]) / (width * height + anchor2[0] * anchor2[1])
    return 0 if iou1 <= iou2 else 1


class Element(object):
    def __init__(self, feature_path, labels_path, grd_depth_map_as_input=False):
        self.feature_path = feature_path[0]
        self.grd_path = feature_path[1]
        self.depth_path = labels_path[0]
        self.obstacles_path = labels_path[1]
        self.grd_depth_map_as_input = grd_depth_map_as_input

    def get_features(self):
        rgb_input = cv.imread(self.feature_path[0], cv.IMREAD_COLOR)
        if self.grd_depth_map_as_input:
            grd_input = cv.imread(self.grd_path[0], cv.IMREAD_GRAYSCALE)
            features = {}
            features["rgb"] = rgb_input
            features["ground"] = np.expand_dims(grd_input, 2)
            return features
        return rgb_input
    
    def get_labels(self):
        depth_label = cv.imread(self.depth_path[0], cv.IMREAD_GRAYSCALE)
        
        with open(self.obstacles_path[0], 'r') as file:
            obstacles = file.readlines()
        obstacles = [line.strip() for line in obstacles]
        
        obstacles_label = np.zeros(shape=(5,8,2,7))
        
        for obstacle in obstacles:
            parsed_str_obs = obstacle.split()
            
            anchor_idx = select_anchor_index(float(parsed_str_obs[4]), float(parsed_str_obs[5]))
            
            col = int(parsed_str_obs[0])
            row = int(parsed_str_obs[1])
            obstacles_label[row, col, anchor_idx, 0] = 1.0 # confidence
            obstacles_label[row, col, anchor_idx, 1] = float(parsed_str_obs[2]) # x
            obstacles_label[row, col, anchor_idx, 2] = float(parsed_str_obs[3]) # y
            obstacles_label[row, col, anchor_idx, 3] = float(parsed_str_obs[4]) # w
            obstacles_label[row, col, anchor_idx, 4] = float(parsed_str_obs[5]) # h
            obstacles_label[row, col, anchor_idx, 5] = float(parsed_str_obs[6]) * 0.1 # m
            obstacles_label[row, col, anchor_idx, 6] = float(parsed_str_obs[7]) / 100. # v
        
        labels = {}
        labels["depth"] = np.expand_dims(depth_label, 2)
        labels["obstacles"] = obstacles_label
        return labels