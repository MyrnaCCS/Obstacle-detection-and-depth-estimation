import numpy as np

class Obstacle(object):
	def __init__(self, x, y, w, h, obs_stats=None, conf_score=None):
		self.x = int(x)
		self.y = int(y)
		self.w = int(w)
		self.h = int(h)
		self.confidence = conf_score
		self.valid_points = -1
		if obs_stats is not None:
			self.depth_mean = obs_stats[0]
			self.depth_variance = obs_stats[1]
		else:
			self.depth_mean = -1
			self.depth_variance = -1
		

	def compute_depth_stats(self, depth):
		if len(depth.shape) == 4:
			roi_depth = depth[0, self.y:self.y+self.h, self.x:self.x+self.w, 0]
		else:
			roi_depth = depth[self.y:self.y+self.h, self.x:self.x+self.w]

		mean_depth = 0
		squared_sum = 0
		valid_points = 0

		for y in range(0, self.h):
			for x in range(0, self.w):
				if roi_depth[y,x] < 3.0 and roi_depth[y,x] > 0.0:
					mean_depth += roi_depth.item(y, x)
					squared_sum += roi_depth.item(y, x)**2
					valid_points += 1

		if valid_points > 0:
			mean_depth /= valid_points
			var_depth = (squared_sum / valid_points) - (mean_depth**2)
		else:
			mean_depth = -1
			var_depth = -1
		
		return mean_depth, var_depth, valid_points


	def evaluate_estimation(self, estimated_depth):
		estimated_mean, estimated_var, valid_points = self.compute_depth_stats(estimated_depth)
		mean_rmse = (self.depth_mean - estimated_mean)**2
		mean_variance = (self.depth_variance - estimated_var)**2
		return np.sqrt(mean_rmse + 1e-6), np.sqrt(mean_variance + 1e-6), valid_points

	
	def compute_depth_stats_from_estimation(self, depth):
		self.x = max(0, self.x)
		self.y = max(0, self.y)
		self.w = min(256-self.x, self.w)
		self.h = min(160-self.y, self.h)
		self.depth_mean, self.depth_variance, self.valid_points = self.compute_depth_stats(depth)