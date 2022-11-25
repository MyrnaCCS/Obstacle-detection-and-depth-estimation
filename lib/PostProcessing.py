import numpy as np
from EvaluationUtils import iou

def non_maximal_suppresion(obstacles_list, iou_thresh=0.7):
	#Flag: is one if is a valid detection
	valid_detection = np.ones(shape=len(obstacles_list), dtype=np.uint8)
	n = len(obstacles_list)
	for i in range(n-1):
		obstacle_1 = obstacles_list[i]
		for j in range(i+1, n):
			#Compute IOU(obstacle_1, obstacle_2)
			obstacle_2 = obstacles_list[j]
			iou = iou((obstacle_1.x, obstacle_1.y, obstacle_1.w, obstacle_1.h),
					 (obstacle_2.x, obstacle_2.y, obstacle_2.w, obstacle_2.h))
			if iou > iou_thresh:
				#Select the best detection
				if obstacle_1.confidence > obstacle_2.confidence:
					valid_detection[j] = 0
				elif obstacle_1.confidence < obstacle_2.confidence:
					valid_detection[i] = 0
	#As a result: list with no multiple detections
	best_detections_list = []
	for i in range(n):
		flag = valid_detection[i]
		if flag == 1:
			best_detections_list.append(obstacles_list[i])
	return best_detections_list

def compute_correction_factor(depth, obstacles):
	mean_corr = 0
	it = 0
	for obstacle in obstacles:
		top = max(obstacle.y, 0)
		bottom = min(obstacle.y + obstacle.h, depth.shape[1])
		left = max([obstacle.x, 0])
		right = min(obstacle.x + obstacle.w, depth.shape[2])
		depth_roi = depth[0, top:bottom, left:right, 0]
		if depth_roi:
			mean_corr += obstacle.depth_mean / np.mean(depth_roi)
			it += 1
	# average factor
	if it > 0:
		mean_corr /= it
	else:
		mean_corr = 1.
	return mean_corr


def compute_correction_factor_depth_ground(depth, ground_depth_map, obstacles):
    mean_corr = 0
    it = 0
    for obstacle in obstacles:
        bottom = min(obstacle.y + obstacle.h, depth.shape[1]-1)
        center = obstacle.x + (obstacle.w/2)
        if center > depth.shape[2]/2:
            corner = np.max([obstacle.x-5, 0])
        else:
            corner = np.min([obstacle.x+obstacle.w+5, depth.shape[2]-1])
        if ground_depth_map[bottom, corner] < 4 and obstacle.w < 240:
            mean_corr += ground_depth_map[bottom, corner] / depth[0, bottom, corner, 0]
            it += 1
    # average factor
    if it > 0:
        mean_corr /= it
    else:
        mean_corr = 1.0
    return mean_corr