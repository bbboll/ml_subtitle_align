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

	def train_model(self, input_3d):
		"""
		"""
		return self._create_model(input_3d, is_training=True)

	def test_model(self, input_3d):
		"""
		"""
		return self._create_model(input_3d, is_training=False)

	def load_variables_from_checkpoint(self, session, checkpoint):
		"""
		"""
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(session, checkpoint)

	def _create_model(self, input_3d, is_training = True):
		"""
		"""
		if is_training:
			dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
		input_4d = tf.expand_dims(input_3d, -1)
		first_filter_width = 8
		first_filter_height = 20
		first_filter_count = 64
		first_weights = tf.Variable(
			tf.truncated_normal([
				first_filter_height,
				first_filter_width,
				1,
				first_filter_count
			], stddev=0.01)
		)
		first_bias = tf.Variable(tf.zeros([first_filter_count]))
		first_conv = tf.nn.conv2d(input_4d, first_weights, [1, 1, 1, 1], "SAME") + first_bias
		first_relu = tf.nn.relu(first_conv)
		if is_training:
			first_dropout = tf.nn.dropout(first_relu, dropout_prob)
		else:
			first_dropout = first_relu
		max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
		second_filter_width = 4
		second_filter_height = 10
		second_filter_count = 64
		second_weights = tf.Variable(
			tf.truncated_normal([
				second_filter_height,
				second_filter_width,
				first_filter_count,
				second_filter_count
			], stddev=0.01)
		)
		second_bias = tf.Variable(tf.zeros([second_filter_count]))
		second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], "SAME") + second_bias
		second_relu = tf.nn.relu(second_conv)
		if is_training:
			second_dropout = tf.nn.dropout(second_relu, dropout_prob)
		else:
			second_dropout = second_relu
		second_conv_shape = second_dropout.get_shape()
		second_conv_output_width = second_conv_shape[2]
		second_conv_output_height = second_conv_shape[1]
		second_conv_element_count = int(second_conv_output_width * second_conv_output_height * second_filter_count)
		flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])
		label_count = self.config["label_count"]
		final_fc_weights = tf.Variable(
			tf.truncated_normal([
				second_conv_element_count,
				label_count
			], stddev=0.01)
		)
		final_fc_bias = tf.Variable(tf.zeros([label_count]))
		final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
		if is_training:
			return final_fc, dropout_prob
		return final_fc


if __name__ == "__main__":
	pass
