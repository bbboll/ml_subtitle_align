import numpy as np
import tensorflow as tf
from .base_model import BaseModel
from .base_model import OUTPUT_DIM

class Model(BaseModel):

	def _create_model(self, input_3d, is_training=True):
		"""
		Model structure:
		Input -> Conv2DLSTM -> Dense -> Dense -> Ouput
		The embedding of conv LSTM output into the label space
		is performed by two dense layers.
		"""
		if is_training:
			# load hyperparameter for dropout layers
			# lower keep probability can be used to control overfitting
			keep_prob = tf.placeholder(tf.float32, name="keep_prob")

		# prepare input for convolution
		input_4d = tf.expand_dims(input_3d, -1)

		# TODO: 
		# https://www.tensorflow.org/tutorials/recurrent
		# with cells of type https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/Conv2DLSTMCell

		# flatten convolutional output
		third_conv_shape = third_dropout.get_shape()
		third_conv_output_width = third_conv_shape[2]
		third_conv_output_height = third_conv_shape[1]
		third_conv_element_count = int(third_conv_output_width * third_conv_output_height * third_filter_count)
		flattened_third_conv = tf.reshape(third_dropout, [-1, third_conv_element_count])

		# encode convolution output into label space
		dense_layer = tf.layers.dense(flattened_third_conv, 1000, activation=tf.nn.sigmoid)
		final_fc = tf.layers.dense(dense_layer, OUTPUT_DIM, activation=tf.nn.sigmoid)

		if is_training:
			return final_fc, keep_prob
		return final_fc