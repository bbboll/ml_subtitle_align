import numpy as np
import os
import os.path
import tensorflow as tf

def _get_full_path(*rel_path):
	"""Make absolute path to a file or directory in the project folder ml_subtitle_align.

	Arguments:
        *rel_path: List of path elements.

    Returns:
        `str`: Absolute path to requested file or directory.
	"""
	path = os.path.abspath(__file__) # `.../ml_subtitle_align/model/model.py`
	path = os.path.dirname(path) # `.../ml_subtitle_align/model/`
	path = os.path.dirname(path) # `.../ml_subtitle_align/`
	return os.path.join(path, *rel_path)

OUTPUT_DIM = 1500

class Model(object):
	"""

	Attributes:
	"""

	def __init__(self):
		"""Initialize a new empty object.
		"""
		self.config = None

	def set_config(self, **arguments):
		"""
		"""
		self.config = arguments

	def train_model(self, input_3d, dense=True):
		"""
		"""
		return self._create_model(input_3d, is_training=True, dense=dense)

	def test_model(self, input_3d, dense=True):
		"""
		"""
		return self._create_model(input_3d, is_training=False, dense=dense)

	def load_variables_from_checkpoint(self, session, checkpoint):
		"""
		"""
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(session, checkpoint)

	def _create_model(self, input_3d, is_training=True, dense=True):
		"""
		Model structure:
		Input -> Conv2d -> relu activation (-> dropout) -> pooling
		      -> Conv2d -> relu activation (-> dropout)
		Two convolutional layers are used to perform translation-invariant classification.
		If the model is set to be dense, the embedding of covolution output into the label
		space is performed by two dense layers.
		"""
		if is_training:
			# load hyperparameter for dropout layers
			# lower keep probability can be used to control overfitting
			keep_prob = tf.placeholder(tf.float32, name="keep_prob")

		# prepare input for convolution
		input_4d = tf.expand_dims(input_3d, -1)

		# convolution 1 parameters
		first_filter_width  = 8
		first_filter_height = 20
		first_filter_count  = 64
		first_weights = tf.Variable(
			tf.truncated_normal([
				first_filter_height,
				first_filter_width,
				1,
				first_filter_count
			], stddev=0.01)
		)

		# convolution 1
		first_bias = tf.Variable(tf.zeros([first_filter_count]))
		first_conv = tf.nn.conv2d(input_4d, first_weights, [1, 1, 1, 1], "SAME") + first_bias

		# activation (+ dropout) + pooling for convolution 1
		first_relu = tf.nn.relu(first_conv)
		if is_training:
			first_dropout = tf.nn.dropout(first_relu, keep_prob)
		else:
			first_dropout = first_relu
		max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

		# convolution 2 parameters
		second_filter_width  = 4
		second_filter_height = 10
		second_filter_count  = 64
		second_weights = tf.Variable(
			tf.truncated_normal([
				second_filter_height,
				second_filter_width,
				first_filter_count,
				second_filter_count
			], stddev=0.01)
		)
		
		# convolution 2
		second_bias = tf.Variable(tf.zeros([second_filter_count]))
		second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], "SAME") + second_bias

		# activation (+ dropout) for convolution 2
		second_relu = tf.nn.relu(second_conv)
		if is_training:
			second_dropout = tf.nn.dropout(second_relu, keep_prob)
		else:
			second_dropout = second_relu
		second_max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

		# convolution 2 parameters
		third_filter_width  = 4
		third_filter_height = 10
		third_filter_count  = 64
		third_weights = tf.Variable(
			tf.truncated_normal([
				third_filter_height,
				third_filter_width,
				first_filter_count,
				third_filter_count
			], stddev=0.01)
		)

		# convolution 3
		third_bias = tf.Variable(tf.zeros([third_filter_count]))
		third_conv = tf.nn.conv2d(second_max_pool, third_weights, [1, 1, 1, 1], "SAME") + third_bias

		# activation (+ dropout) for convolution 2
		third_relu = tf.nn.relu(third_conv)
		if is_training:
			third_dropout = tf.nn.dropout(third_relu, keep_prob)
		else:
			third_dropout = third_relu

		# flatten convolutional output
		third_conv_shape = third_dropout.get_shape()
		third_conv_output_width = third_conv_shape[2]
		third_conv_output_height = third_conv_shape[1]
		third_conv_element_count = int(third_conv_output_width * third_conv_output_height * third_filter_count)
		flattened_third_conv = tf.reshape(third_dropout, [-1, third_conv_element_count])

		# encode convolution output into label space
		if dense:
			dense_layer = tf.layers.dense(flattened_third_conv, 1000, activation=tf.nn.sigmoid)
			final_fc = tf.layers.dense(dense_layer, OUTPUT_DIM, activation=tf.nn.sigmoid)
		else:
			final_fc_weights = tf.Variable(
				tf.truncated_normal([
					second_conv_element_count,
					OUTPUT_DIM
				], stddev=0.01)
			)
			final_fc_bias = tf.Variable(tf.zeros([OUTPUT_DIM]))
			final_fc = tf.matmul(flattened_third_conv, final_fc_weights) + final_fc_bias

		if is_training:
			return final_fc, keep_prob
		return final_fc


if __name__ == "__main__":
	pass
