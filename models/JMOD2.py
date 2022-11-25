import os
import sys
import time
import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Reshape, Convolution2D, Input, Conv2DTranspose, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

from lib.Dataset import Dataset
from lib.DepthLossFunction import *
from lib.DetectionLossFunction import *
from lib.DataAugmentation import DataAugmentationStrategy
from lib.EvaluationUtils import get_detected_obstacles_from_detector_v2
from lib.PostProcessing import *

class JMOD2(object):
    def __init__(self, config):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_memory_fraction
        tf_config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=tf_config))
        self.config = config
        self.dataset = Dataset(self.config)
        self.data_augmentation_strategy = DataAugmentationStrategy()
        self.training_data = {}
        self.validation_data = {}
        self.model = self.build_model()
    
    
    def load_data(self):
        self.dataset.read_data()
        self.training_data = self.dataset.generate_data()
        
        # Split into validation and training
        np.random.shuffle(self.training_data)
        train_val_split_idx = int(len(self.training_data)*(1-self.config.validation_split))
        self.validation_data = self.training_data[train_val_split_idx:]
        self.training_data = self.training_data[:train_val_split_idx]
    
    
    def build_model(self):
        input = Input(shape=(self.config.input_height, self.config.input_width, self.config.input_channel), name="input")
        vgg19 = VGG19(include_top=False, weights=None, input_tensor=input, input_shape=(self.config.input_height, self.config.input_width, self.config.input_channel))
        
        vgg19.layers.pop()
        
        #Depth section
        x = vgg19.layers[-1].output
        for num_filters in [128, 64, 32, 16]:
            x = Conv2DTranspose(num_filters, (4, 4), padding="same", strides=(2, 2))(x)
            x = PReLU()(x)
        depth_output = Convolution2D(1, (5, 5), padding="same", activation="relu", name="depth_output")(x)
        
        #Detection section
        x = vgg19.layers[-1].output
        x = MaxPooling2D(pool_size=(2, 2), name='det_maxpool')(x)
        for num_filters in [512, 1024, 512, 1024, 512]:
            if num_filters == 512:
                x = Convolution2D(num_filters, (3, 3), activation="relu", padding="same")(x)
            else:
                x = Convolution2D(num_filters, (1, 1), activation="relu", padding="same")(x)
        x = Convolution2D(14, (1, 1), activation="linear", padding="same")(x)
        detection_output = Reshape((5, 8, 2, 7), name='detection_output')(x)
        
        #Model
        jmod2_model = Model(inputs=input, outputs=[depth_output, detection_output])
        jmod2_model.compile(loss={'depth_output':log_normals_loss, 'detection_output':yolo_v2_loss},
							optimizer=Adam(lr=self.config.learning_rate, clipnorm=1.0),
							metrics={'depth_output': [rmse_metric, logrmse_metric, sc_inv_logrmse_metric], 
									 'detection_output': [iou_metric, recall, precision, yolo_objconf_loss, yolo_nonobjconf_loss, yolo_xy_loss, yolo_wh_loss, yolo_mean_loss, yolo_var_loss]},
							loss_weights=[5.0, 1.0])
        return jmod2_model
    
    
    def prepare_data_for_model(self, features, label):
        features = np.asarray(features)
        features = features.astype('float32')
        features /= 255.0
        
        # Prepare output : lista de numpy arrays
        labels_depth = np.zeros(shape=(features.shape[0], features.shape[1], features.shape[2], 1), dtype=np.float32) # Gray Scale
        labels_obs = np.zeros(shape=(features.shape[0], 5, 8, 2, 7), dtype=np.float32) # Obstacle output
        for idx, element in enumerate(label):
            labels_depth[idx, ...] = np.asarray(element["depth"]).astype(np.float32) / 255.0
            labels_obs[idx, ...] = np.asarray(element["obstacles"]).astype(np.float32)
        
        return features, [labels_depth, labels_obs]
    
    
    def train_data_generator(self):
        np.random.shuffle(self.training_data)
        curr_batch = 0
        self.training_data = list(self.training_data)
        while True:
            if (curr_batch + 1) * self.config.batch_size > len(self.training_data):
                np.random.shuffle(self.training_data)
                curr_batch = 0
            x_train = []
            y_train = []
            for sample in self.training_data[curr_batch * self.config.batch_size: (curr_batch + 1) * self.config.batch_size]:
                # Get input
                features = sample.get_features()
                # Get output
                label = sample.get_labels()
                if self.data_augmentation_strategy is not None:
                    features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=False)
                # Append in batch list
                x_train.append(features)
                y_train.append(label)
            x_train, y_train = self.prepare_data_for_model(x_train, y_train)
            curr_batch += 1
            yield x_train , y_train
    
    
    def validation_data_generator(self):
        np.random.shuffle(self.validation_data)
        curr_batch = 0
        while True:
            if (curr_batch + 1) * self.config.batch_size > len(self.validation_data):
                np.random.shuffle(self.validation_data)
                curr_batch = 0
            x_train = []
            y_train = []
            for sample in self.validation_data[curr_batch * self.config.batch_size: (curr_batch + 1) * self.config.batch_size]:
                features = sample.get_features()
                label = sample.get_labels()
                if self.data_augmentation_strategy is not None:
                    features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=True)
                x_train.append(features)
                y_train.append(label)
            x_train, y_train = self.prepare_data_for_model(x_train, y_train)
            curr_batch += 1
            yield x_train , y_train
    
    
    def train(self, initial_epoch=0):
        # Save model summary
        orig_stdout = sys.stdout
        f = open(os.path.join(self.config.model_dir, "model_summary.txt"), "w")
        sys.stdout = f
        
        # Print layers in model summary.txt
        for layer in self.model.layers:
            print(layer.get_config())
        sys.stdout = orig_stdout
        f.close()
        
        # Save img model summaty
        plot_model(self.model, show_shapes=True, to_file=os.path.join(self.config.model_dir, "model_structure.png"))
        
        # Inicial time
        t0 = time.time()
        
        # Samples per epoch
        samples_per_epoch = int(np.floor(len(self.training_data) / self.config.batch_size))
        
        # Validation steps
        val_step = int(np.floor(len(self.validation_data) / self.config.batch_size))
        print("Samples per epoch: {}".format(samples_per_epoch))
        
        # Callbacks
        model_checkpoint = ModelCheckpoint(os.path.join(self.config.model_dir, 'weights-{epoch:02d}-{loss:.2f}.hdf5'),
                                           monitor="loss", verbose=2, save_best_only=False, save_weights_only=False, 
                                           mode="auto", period=self.config.log_step)
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
        
        # Train
        history = self.model.fit_generator(generator=self.train_data_generator(),
                                           steps_per_epoch=samples_per_epoch,
                                           callbacks=[model_checkpoint, es],
                                           validation_data=self.validation_data_generator(),
                                           validation_steps=val_step,
                                           epochs=self.config.num_epochs,
                                           verbose=2,
                                           initial_epoch=initial_epoch)
        # Final time
        t1 = time.time()
        print("Training completed in " + str(t1 - t0) + " seconds")
        
        return history
    

    def resume_training(self, weights_file, initial_epoch):
        self.model.load_weights(weights_file)
        history = self.train(initial_epoch)
        return history
    
    
    def load_model(self):
        self.model.load_weights("weights/jmod2.hdf5")
    

    def run(self, input, ground_plane_depth_map=None):
        mean = np.load('Indoors_RGB_mean.npy')
        
        if len(input.shape) == 2:
            input = np.expand_dims(input, axis=-1)
        
        if input.shape[2] == 1:
            input = np.tile(input, 3)
        
        if len(input.shape) == 3:
            input = np.expand_dims(input - mean/255.0, 0)
        else:
            input[0, ...] -= mean/255.0
        
        # Prediction
        t0 = time.time()
        
        net_output = self.model.predict(input)
        
        print ("Elapsed time: {}").format(time.time() - t0)
        
        # Depth map
        pred_depth = net_output[0] * 7.0
        
        # Obstacles
        pred_detection = net_output[1]
        pred_obstacles = get_detected_obstacles_from_detector_v2(pred_detection, self.config.detector_confidence_thr)
        
        # Depth map corrected
        if ground_plane_depth_map is not None:
            correction_factor = compute_correction_factor_depth_ground(pred_depth, ground_plane_depth_map, pred_obstacles)
        else:
            correction_factor = compute_correction_factor(pred_depth, pred_obstacles)
        corrected_depth = np.array(pred_depth) * correction_factor

        #Eliminate multiple detections
        if self.config.non_max_suppresion:
            pred_obstacles = non_maximal_suppresion(pred_obstacles, iou_thresh=0.3)
        
        return [pred_depth, pred_obstacles, corrected_depth]