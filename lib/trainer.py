import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use('Agg')


class Trainer(object):
	def __init__(self, config, model, rng):
		self.config = config
		self.model = model
		self.model.load_data()

	def train(self):
		print("[*] Training starts...")
		history = self.model.train()
		self.plot_metrics(history)
	
	def resume_training(self):
		print("Resuming training from weights file: ", self.config.weights_path)
		history = self.model.resume_training(self.config.weights_path, self.config.initial_epoch)
		self.plot_metrics(history)

	def plot_metrics(self, history):
		plt.ioff()
		name_metrics = ['detection_output_iou_metric', 'detection_output_recall', 'detection_output_precision', 
						'depth_output_rmse_metric', 'depth_output_logrmse_metric', 'depth_output_sc_inv_logrmse_metric',
						'loss', 'depth_output_loss', 'detection_output_loss',
						'detection_output_yolo_objconf_loss', 'detection_output_yolo_nonobjconf_loss', 
						'detection_output_yolo_xy_loss', 'detection_output_yolo_wh_loss', 
						'detection_output_yolo_mean_loss', 'detection_output_yolo_var_loss']
		
		title_metrics = ['Detection IoU', 'Detection Recall', 'Detection Precision', 
						 'Depth RMSE (linear)', 'Depth RMSE (log)', 'Depth RMSE (log, scale-invariant)',
						 'Loss', 'Depth Loss', 'Detection Loss',
						 'Detection Loss Confidence (obstacle)', 'Detection Loss Confidence (no obstacle)',
						 'Detection Loss Location', 'Detection Loss Size',
						 'Detection Loss Mean Depth', 'Detection Loss Variance Depth']
		
		label_metrics = ['IoU', 'Recall', 'Precision', 
						 'RMSE', 'RMSE (log)', 'RMSE (log, scale-invariant)',
						 'Loss', 'Loss', 'Loss',
						 'Loss', 'Loss', 
						 'Loss', 'Loss',
						 'Loss', 'Loss']
		
		path_figure = ['iou.pdf', 'recall.pdf', 'precision.pdf', 
					   'rmse.pdf', 'logrmse.pdf', 'invlogrmse.pdf',
					   'loss.pdf', 'lossdepth.pdf', 'lossdetection.pdf',
					   'yolo_loss_conf.pdf', 'yolo_loss_conf_noobj.pdf',
					   'yolo_xy_loss.pdf', 'yolo_wh_loss.pdf',
					   'yolo_mean_loss.pdf', 'yolo_var_loss.pdf']
		
		for name, title, label_y, path in zip(name_metrics, title_metrics, label_metrics, path_figure):
			fig = plt.figure()
			plt.plot(history.history[name], label = 'Training value', color = 'darkslategray')
			plt.plot(history.history['val_'+name], label = 'Validation value', color = 'darkslategray', linestyle = '--')
			plt.title(title)
			plt.xlabel('Epochs')
			plt.ylabel(label_y)
			plt.legend()
			plt.savefig(os.path.join('graphics', path))
			plt.close(fig)

		# convert the history.history dict to a pandas DataFrame:     
		hist_df = pd.DataFrame(history.history)

		# save to csv: 
		hist_csv_file = 'graphics/history.csv'
		with open(hist_csv_file, mode='w') as f:
			hist_df.to_csv(f)