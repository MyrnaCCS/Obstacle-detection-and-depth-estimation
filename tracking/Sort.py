import numpy as np
from KalmanFilter import KalmanBoxTracker
from SortUtils import associate_detections_to_trackers
from lib.Obstacle import Obstacle


class Sort(object):
    def __init__(self, max_age, min_hits, iou_threshold):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracker_list = []
        self.frame_count = 0

    
    def update(self, obstacles):
        self.frame_count += 1

        # create detections array from obstacle list
        detections = np.zeros(shape=(len(obstacles), 5))
        for idx, obstacle in enumerate(obstacles):
            detections[idx] = [obstacle.x, obstacle.y, obstacle.x+obstacle.w, obstacle.y+obstacle.h, obstacle.confidence]
        
        # get predicted locations from existing trackers.
        prediction_of_trackers = []
        nan_trackers = []
        
        for idx, tracker in enumerate(self.tracker_list):
            box = tracker.predict()[0]
            if any(np.isnan(box)):
                nan_trackers.append(idx)
            else:
                prediction_of_trackers.append([box[0], box[1], box[2], box[3], 0])
        
        prediction_of_trackers = np.asarray(prediction_of_trackers)
        
        for idx in nan_trackers:
            del self.tracker_list[idx]
        
        matched_list, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, prediction_of_trackers, self.iou_threshold)

        # update matched trackers with assigned detections
        for match in matched_list:
            self.tracker_list[match[1]].update(detections[match[0], :])

        # create and initialise new trackers for unmatched detections
        for idx in unmatched_detections:
            new_tracker = KalmanBoxTracker(detections[idx, :])
            self.tracker_list.append(new_tracker)
        
        ret = []
        obstacles_ret = []
        
        for idx, tracker in enumerate(self.tracker_list):
                current_state = tracker.get_state()[0]
                if (tracker.time_since_update <= self.max_age) and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                    ret.append(np.concatenate((current_state, [tracker.id+1])).reshape(1, -1)) # +1 as MOT benchmark requires positive
                    obstacles_ret.append(Obstacle(current_state[0], current_state[1], current_state[2]-current_state[0], current_state[3]-current_state[1]))
                # remove dead tracker
                if(tracker.time_since_update > self.max_age):
                    del self.tracker_list[idx]
        
        if ret:
            return np.concatenate(ret), obstacles_ret
        
        return np.empty((0,5)), obstacles_ret