import numpy as np

class DataAugmentationStrategy(object):

	def process_sample_specialized(self, features):
		aug_features = features * np.random.normal(loc=1.0, scale=0.05, size=features.shape)
		if (np.random.rand() > 0.85):
			aug_features[..., 0] =  aug_features[..., 0] * np.random.normal(loc=1.0, scale=0.1, size=features[..., 0].shape)
		if (np.random.rand() > 0.85):
			aug_features[..., 1] =  aug_features[..., 1] * np.random.normal(loc=1.0, scale=0.1, size=features[..., 0].shape)
		if (np.random.rand() > 0.85):
			aug_features[..., 2] = aug_features[..., 2] * np.random.normal(loc=1.0, scale=0.1, size=features[..., 0].shape)
		return aug_features
		
	def process_sample(self, features, label, is_test=False):
		if is_test is False:
			aug_features = self.process_sample_specialized(features)
			return aug_features, label
		else:
			return features, label
