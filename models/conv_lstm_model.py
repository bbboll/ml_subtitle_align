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
		time_seq_len = 1
		output_cannels = 1

		if is_training:
			# load hyperparameter for dropout layers
			# lower keep probability can be used to control overfitting
			keep_prob = tf.placeholder(tf.float32, name="keep_prob")

		# prepare input for convolution
		input_shape = input_3d.get_shape()
		input_4d = tf.reshape(input_3d, [-1, time_seq_len, int(input_shape[1]), int(input_shape[2])])
		input_4d = tf.expand_dims(input_4d, -1)

		# convolutional lstm
		conv_lstm_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[int(input_shape[1]), int(input_shape[2]), 1],
													   kernel_shape=[3, 3],
													   output_channels=output_cannels)
		conv_lstm_output, conv_lstm_state = tf.nn.dynamic_rnn(conv_lstm_cell,
															  input_4d,
															  time_major=False,
															  dtype=tf.float32)

		conv_output = conv_lstm_state[0]

		# flatten convolutional output
		conv_shape = conv_output.get_shape()
		conv_output_width = conv_shape[2]
		conv_output_height = conv_shape[1]
		conv_element_count = int(conv_output_width * conv_output_height)
		flattened_conv = tf.reshape(conv_output, [-1, conv_element_count])

		# encode convolution output into label space
		dense_layer = tf.layers.dense(flattened_conv, 1000, activation=tf.nn.sigmoid)
		final_fc = tf.layers.dense(dense_layer, OUTPUT_DIM, activation=tf.nn.sigmoid)

		if is_training:
			return final_fc, keep_prob
		return final_fc
