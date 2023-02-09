import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
	return v.lower() in ('true', '1')

def add_argument_group(name):
	arg = parser.add_argument_group(name)
	arg_lists.append(arg)
	return arg

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset_path', type=str, default='data/IndoorsDataset')
data_arg.add_argument('--data_main_dir', type=str, default='')
data_arg.add_argument('--data_train_val_path', type=eval, nargs='+', default=['40_D', '41_D', '44_D', '45_D', '46_D', '47_D', '48_D', '49_D', '50_D', '52_D', '54_D', '55_D', '56_D', '57_D'])
data_arg.add_argument('--data_test_path', type=eval, nargs='+', default=['42_D', '43_D', '51_D', '53_D'])
data_arg.add_argument('--input_height', type=int, default=160)
data_arg.add_argument('--input_width', type=int, default=256)
data_arg.add_argument('--input_channel', type=int, default=3)
data_arg.add_argument('--img_extension', type=str, default="png")

#JMOD2 param
jmod2_arg = add_argument_group('JMOD2')
jmod2_arg.add_argument('--detector_confidence_thr', type=int, default=0.5)
jmod2_arg.add_argument('--non_max_suppresion', type=str2bool, default=False)

# Training / Testing param
train_arg = add_argument_group('Training')
train_arg.add_argument('--exp_name', type=str, default='NAME_OF_EXPERIMENT')
train_arg.add_argument('--validation_split', type=float, default=0.2, help='')
train_arg.add_argument('--batch_size', type=int, default=64, help='')
train_arg.add_argument('--num_epochs', type=int, default=100, help='')
train_arg.add_argument('--learning_rate', type=float, default=1e-4, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--weights_path_vgg19', type=str, default="weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")# Needed to train on cluster
train_arg.add_argument('--weights_path', type=str, default="weights/jmod2.hdf5")# Used for finetuning or to resume training
train_arg.add_argument('--resume_training', type=str2bool, default=False)
train_arg.add_argument('--initial_epoch', type=int, default=100)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=20, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')# Directory where to save model checkpoints
misc_arg.add_argument('--debug', type=str2bool, default=True)
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=0.8)
misc_arg.add_argument('--max_image_summary', type=int, default=4)

# Sort
sort_arg = add_argument_group('Sort')
sort_arg.add_argument('--tracking', type=str2bool, default=False)
sort_arg.add_argument("--max_age", help="Max number of frames to keep alive a track without associated detections.", type=int, default=3)
sort_arg.add_argument("--min_hits", help="Minimum number of associated detections before track is initialised.", type=int, default=0)
sort_arg.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)

def get_config():
	config, unparsed = parser.parse_known_args()
	return config, unparsed