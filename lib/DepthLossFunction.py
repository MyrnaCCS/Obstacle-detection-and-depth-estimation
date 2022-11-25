import tensorflow as tf
import keras.backend as K
import numpy as np
from config import get_config

def rmse_metric(y_true, y_pred):
	rmse = K.sqrt(K.mean(K.square((y_true - y_pred))) + K.epsilon())
	return rmse

def logrmse_metric(y_true, y_pred):
	first_log = K.log(y_pred + 1.)
	second_log = K.log(y_true + 1.)
	return K.sqrt(K.mean(K.mean(K.square(first_log - second_log), axis=0)) + K.epsilon())

def sc_inv_logrmse_metric(y_true, y_pred):
	first_log = K.log(y_pred + 1.)
	second_log = K.log(y_true + 1.)
	sc_inv_term = K.square(K.mean(K.mean((first_log - second_log), axis=-1)))
	log_term = K.sqrt(K.mean(K.mean(K.square(first_log - second_log), axis=0)) + K.epsilon())
	return log_term - sc_inv_term

def log_normals_loss(y_true, y_pred):
    config, unparsed = get_config()

    # First terms
    first_log = K.log(y_pred + 1.0)
    second_log = K.log(y_true + 1.0)
    log_term = K.sqrt(K.mean(K.square(first_log - second_log), axis=-1) + K.epsilon())
    sc_inv_term = K.square(K.mean((first_log - second_log), axis=-1))
    
    w_x = K.variable(np.array([[-1.0, 0.0, 1.0],
                               [-1.0, 0.0, 1.0],
                               [-1.0, 0.0, 1.0]]).reshape(3, 3, 1, 1))

    w_y = K.variable(np.array([[-1.0, -1.0, -1.0],
                               [0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0]]).reshape(3, 3, 1, 1))
    
    #Truth Normals
    dzdx_true = K.conv2d(y_true, w_x, padding="same")
    dzdy_true = K.conv2d(y_true, w_y, padding="same")
    norm_magnitude = K.sqrt(K.pow(dzdx_true, 2) + K.pow(dzdy_true, 2) + 1.0)
    n3 = 1.0 / norm_magnitude
    n1 = - dzdx_true / norm_magnitude
    n2 = - dzdy_true / norm_magnitude
    norm = K.concatenate(tensors=[n1, n2, n3], axis=-1)

    #Gradient Prediction
    dzdx_pred = K.conv2d(y_pred, w_x, padding="same")
    dzdy_pred = K.conv2d(y_pred, w_y, padding="same")
    norm_magnitude_x = K.sqrt(K.pow(dzdx_pred, 2) + 1.0)
    norm_magnitude_y = K.sqrt(K.pow(dzdy_pred, 2) + 1.0)
    batch_size = config.batch_size
    grad_x = K.concatenate(tensors=[K.constant(1.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]]) / norm_magnitude_x,
                                    K.constant(0.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]]) / norm_magnitude_x, dzdx_pred / norm_magnitude_x], axis=-1)
    grad_y = K.concatenate(tensors=[K.constant(0.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]]) / norm_magnitude_y,
                                    K.constant(1.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]]) / norm_magnitude_y, dzdy_pred / norm_magnitude_y], axis=-1)

    #Last term
    dot_term_x = K.sum(norm[...] * grad_x[...], axis=-1, keepdims=True)
    dot_term_y = K.sum(norm[...] * grad_y[...], axis=-1, keepdims=True)
    norm_term = K.mean(K.square(dot_term_x), axis=-1) + K.mean(K.square(dot_term_y), axis=-1)

    loss = log_term - (0.5 * sc_inv_term) + norm_term

    return loss