import sys
sys.path.append('../')
import numpy as np
import cv2
import math
import os

from Obstacle import Obstacle


def get_detected_obstacles_from_detector_v2(prediction, confidence_thr=0.5):
	def sigmoid(x):
		return 1 / (1 + math.exp(-x))

	if len(prediction.shape) == 4:
		prediction = np.expand_dims(prediction, axis=0)

	conf_pred = prediction[0, ..., 0]
	x_pred = prediction[0, ..., 1]
	y_pred = prediction[0, ..., 2]
	w_pred = prediction[0, ..., 3]
	h_pred = prediction[0, ..., 4]
	mean_pred = prediction[0, ..., 5]
	var_pred = prediction[0, ..., 6]

	# img shape
	IMG_WIDTH = 256.
	IMG_HEIGHT = 160.

	# Anchors indoors
	anchors = np.array([[0.21651918, 0.78091232],
						[0.85293483, 0.96561908]], dtype=np.float32)

	# obstacles list
	detected_obstacles = []
	for i in range(0, 5):
		for j in range(0, 8):
			for k in range(0, 2):
				val_conf = sigmoid(conf_pred[i, j, k])
				if val_conf >= confidence_thr:
					x = sigmoid(x_pred[i, j, k])
					y = sigmoid(y_pred[i, j, k])
					w = np.exp(w_pred[i, j, k]) * anchors[k, 0] * IMG_WIDTH
					h = np.exp(h_pred[i, j, k]) * anchors[k, 1] * IMG_HEIGHT
					mean = mean_pred[i, j, k] * 10 * 3.0
					var = var_pred[i, j, k] * 100
					x_top_left = np.floor(((x + j) * 32.) - (w / 2.))
					y_top_left = np.floor(((y + i) * 32.) - (h / 2.))
					detected_obstacles.append(Obstacle(x_top_left, y_top_left, w, h, obs_stats=(mean, var), conf_score=val_conf))

	return detected_obstacles

def get_obstacles_from_file(path_file):
	obstacles_as_string = (open(path_file)).readlines()
	
	obstacle_list = []
	
	for obstacle_string in obstacles_as_string:
		obstacle = obstacle_string.split(' ')
		w = round(float(obstacle[4]) * 256)
		h = round(float(obstacle[5]) * 160)
		x = 32 * (float(obstacle[2]) + int(obstacle[0])) - (w / 2.0)
		y = 32 * (float(obstacle[3]) + int(obstacle[1])) - (h / 2.0)
		mean = float(obstacle[6]) * 3.0
		var = float(obstacle[7])
		obstacle_list.append(Obstacle(max(0, x), max(0, y), w, h, obs_stats=(mean, var), conf_score=1.0))
	
	return obstacle_list


def rmse_error_on_vector(y_true, y_pred):
	mean = np.mean(np.square(y_true - y_pred))
	rmse_error = np.sqrt(mean + 1e-6)
	return rmse_error


def sc_inv_logrmse_error_on_vector(y_true, y_pred):
	first_log = np.log(y_pred + 1.)
	second_log = np.log(y_true + 1.)
	log_term = np.mean(np.square((first_log - second_log)))
	sc_inv_term = np.square(np.mean(first_log - second_log))
	error = log_term - sc_inv_term
	return error


def rmse_log_error_on_matrix(y_true, y_pred):
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	diff = np.square(np.log(y_pred + 1) - np.log(y_true + 1))
	mean = np.mean(diff)
	rmse_error = np.sqrt(mean + 1e-6)
	return rmse_error


def mae_error_on_matrix(y_true, y_pred):
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	error = np.mean(np.abs(y_true - y_pred))
	return error


def rmse_error_on_matrix(y_true, y_pred):
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	rmse_error = rmse_error_on_vector(y_true, y_pred)
	return rmse_error


def sc_inv_logrmse_error_on_matrix(y_true, y_pred):
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	error = sc_inv_logrmse_error_on_vector(y_true, y_pred)
	return error


def compute_obstacle_error_on_depth_branch(estimation, obstacles):
	"Given a depth estimation and a list of obstacles, compute depth error on obstacles"
	obs_area = 0
	obs_m_error = 0
	obs_v_error = 0

	for obstacle in obstacles:
		m_error, v_error, valid_points = obstacle.evaluate_estimation(estimation)
		area = valid_points 
		if m_error != -1: #arbitrary threshold for small obstacles
			obs_area += area
			obs_m_error += m_error * area
			obs_v_error += v_error * area
	
	return obs_m_error, obs_v_error, obs_area


def overlap(l1, r1, l2, r2):
    left = np.where(l1>l2, l1, l2)
    right = np.where(r1>r2, r2, r1)
    return right - left


def iou(box1, box2):
    overlap_w = overlap(box1[0], box1[0]+box1[2], box2[0], box2[0]+box2[2])
    overlap_h = overlap(box1[1], box1[1]+box1[3], box2[1], box2[1]+box2[3])
    # compute intersection
    intersection = max(overlap_w, 0) * max(overlap_h, 0)
    # compute union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    # iou=intersection/union
    iou = intersection / (union + 1e-6)
    return iou


def compute_detection_stats(detected_obstacles, gt_obstacles, iou_thresh=0.5):
	if len(gt_obstacles) > 0:
		closer_gt_obstacles = []

		for det_obstacle in detected_obstacles:
			#Find in GT closer obstacle to the one detected
			max_idx = 0
			max_iou = 0
			for idx, gt_obstacle in enumerate(gt_obstacles):
				iou_score = iou((gt_obstacle.x, gt_obstacle.y, gt_obstacle.w, gt_obstacle.h),
						  		(det_obstacle.x, det_obstacle.y, det_obstacle.w, det_obstacle.h))
				if iou_score > max_iou:
					max_iou = iou_score
					max_idx = idx
			
			closer_gt_obstacles.append((gt_obstacles[max_idx], max_idx, max_iou))
		
		#Result: best iou, depth error, variance error, multiple detections
		iou_for_each_gt_obstacle = np.zeros(shape=len(gt_obstacles), dtype=np.float32)
		depth_error_for_each_gt_obstacle = np.zeros(shape=len(gt_obstacles), dtype=np.float32)
		var_depth_error_for_each_gt_obstacle = np.zeros(shape=len(gt_obstacles), dtype=np.float32)
		n_valid_pred_for_each_gt_obstacle = np.zeros(shape=len(gt_obstacles))

		it = 0
		for elem in closer_gt_obstacles:
			if elem[2] > iou_thresh:
				n_valid_pred_for_each_gt_obstacle[elem[1]] += 1
				if elem[2] > iou_for_each_gt_obstacle[elem[1]]:
					iou_for_each_gt_obstacle[elem[1]] = elem[2]
					depth_error_for_each_gt_obstacle[elem[1]] = rmse_error_on_vector(elem[0].depth_mean, detected_obstacles[it].depth_mean)
					var_depth_error_for_each_gt_obstacle[elem[1]] = rmse_error_on_vector(elem[0].depth_variance, detected_obstacles[it].depth_variance)
				it += 1
		
		n_detected_obstacles = 0
		n_non_detected_obs = 0 #false negatives
		for n in n_valid_pred_for_each_gt_obstacle:
			if n > 0:
				n_detected_obstacles += 1
			else:
				n_non_detected_obs += 1
		
		#Compute average iou, mean error, variance error
		avg_iou = 0
		avg_mean_depth_error = -1
		avg_var_depth_error = -1
		if n_detected_obstacles > 0:
			avg_iou = np.mean(iou_for_each_gt_obstacle[np.nonzero(iou_for_each_gt_obstacle)])
			avg_mean_depth_error = np.mean(depth_error_for_each_gt_obstacle[np.nonzero(depth_error_for_each_gt_obstacle)])
			avg_var_depth_error = np.mean(var_depth_error_for_each_gt_obstacle[np.nonzero(var_depth_error_for_each_gt_obstacle)])
		
		#Compute Precision and Recall
		true_positives = np.sum(n_valid_pred_for_each_gt_obstacle)
		false_positives = len(detected_obstacles) - true_positives
	elif len(detected_obstacles) > 0:
		#detection on image with no gt obstacle
		avg_iou = 0
		avg_mean_depth_error = avg_var_depth_error = -1
		true_positives = 0
		false_positives = len(detected_obstacles)
		n_non_detected_obs = 0
		n_detected_obstacles = 0
	else:
		#detection on image with no gt obstacle, no detections
		avg_iou = -1
		avg_mean_depth_error = avg_var_depth_error = -1
		true_positives = -1
		false_positives = -1
		n_non_detected_obs = -1
		n_detected_obstacles = -1

	return avg_iou, avg_mean_depth_error, avg_var_depth_error, true_positives, false_positives, n_non_detected_obs, n_detected_obstacles


def show_detections(rgb, detection, gt=None, save=True, save_dir=None, file_name=None, print_depths=False, sleep_for=50):
	if len(rgb.shape) == 4:
		rgb = rgb[0, ...]

	if len(rgb.shape) == 3 and rgb.shape[2] == 1:
		rgb = np.tile(rgb, 3)

	if len(rgb.shape) == 2:
		rgb = np.expand_dims(rgb, axis=-1)
		rgb = np.tile(rgb, 3)
	
	output = rgb.copy()
	det_obstacles_data = []
	gt_obstacles_data = []

	for obs in detection:
		cv2.rectangle(output, (obs.x, obs.y), (obs.x+obs.w, obs.y+obs.h), (0,0,255), 2)
		det_obstacles_data.append((obs.x, obs.y, obs.w, obs.h, obs.depth_mean, obs.depth_variance, obs.confidence))
	
	if gt is not None:
		for obs in gt:
			cv2.rectangle(output, (obs.x, obs.y),(obs.x+ obs.w, obs.y+obs.h), (0,255,0), 2)
			gt_obstacles_data.append((obs.x, obs.y, obs.w, obs.h, obs.depth_mean, obs.depth_variance, obs.confidence))
	if save:
		abs_save_dir = os.path.join(os.getcwd(),save_dir)
		if not os.path.exists(os.path.join(abs_save_dir, 'rgb')):
			os.makedirs(os.path.join(abs_save_dir, 'rgb'))
		if not os.path.exists(os.path.join(abs_save_dir, 'detections')):
			os.makedirs(os.path.join(abs_save_dir, 'detections'))

		cv2.imwrite(os.path.join(abs_save_dir, 'rgb', file_name), rgb)
		cv2.imwrite(os.path.join(abs_save_dir, 'detections', file_name), output)

		with open(os.path.join(abs_save_dir,'detections', os.path.splitext(file_name)[0] + '.txt'),'w') as f:
			f.write('Detected obstacles\n')
			for x in det_obstacles_data:
				f.write('x:{},y:{},w:{},h:{},depth:{},var_depth:{},confidence:{}\n'.format(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
			if gt is not None:
				f.write('\nGT obstacles\n')
				for x in gt_obstacles_data:
					f.write('x:{},y:{},w:{},h:{},depth:{},var_depth:{},confidence:{}\n'.format(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
	
	cv2.imshow("Detections(RED:predictions,GREEN: GT", output)
	cv2.waitKey(sleep_for)


def show_depth(rgb, depth, gt=None, save=True, save_dir=None, file_name=None, max_depth=7.5, sleep_for=50):
	if len(rgb.shape) == 4:
		rgb = rgb[0, ...]
	
	if len(depth.shape) == 4:
		depth = depth[0, ...]
	
	if gt is not None and len(gt.shape) == 4:
		gt = gt[0, ...]

	depth_img = np.clip(depth[:, :], 0.0, max_depth)
	depth_img = (depth_img / max_depth * 255.).astype("uint8")
	depth_jet = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

	cv2.imshow("Predicted Depth", depth_jet)

	if gt is not None:
		gt_img = np.clip(gt, 0.0, max_depth)
		gt_img = (gt/max_depth * 255.).astype("uint8")
		gt_jet = cv2.applyColorMap(gt_img, cv2.COLORMAP_JET)
		cv2.imshow("GT Depth", gt_jet)

	if save:
		abs_save_dir = os.path.join(os.getcwd(), save_dir)
		if not os.path.exists(os.path.join(abs_save_dir,'rgb')):
			os.makedirs(os.path.join(abs_save_dir,'rgb'))
		if not os.path.exists(os.path.join(abs_save_dir, 'depth')):
			os.makedirs(os.path.join(abs_save_dir, 'depth'))
		if gt is not None:
			if not os.path.exists(os.path.join(abs_save_dir, 'gt')):
				os.makedirs(os.path.join(abs_save_dir, 'gt'))
			cv2.imwrite(os.path.join(abs_save_dir, 'gt', file_name), gt_jet)
		cv2.imwrite(os.path.join(abs_save_dir, 'rgb', file_name), rgb)
		cv2.imwrite(os.path.join(abs_save_dir, 'depth', file_name), depth_jet)

	cv2.waitKey(sleep_for)