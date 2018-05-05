import tensorflow as tf
import numpy as np
from preprocessing.talk import Talk
import training_routines
import training_data_provider

if __name__ == '__main__':
	run = "run_2018-04-23-10_b70a180695837f53476dce140e1694f9"
	model_load_checkpoint, training_config = training_routines.get_model_from_run(run)
	data_provider = training_data_provider.DataProvider(training_config)

	# start a new tensorflow session
	sess = tf.InteractiveSession()
	input_3d = tf.placeholder(tf.float32, [None, 399, 13], name="input_3d")

	# load model
	model = training_routines.get_model_obj_from_config(training_config)
	prediction = model.test_model(input_3d)
	print("Loading model from checkpoint {}".format(model_load_checkpoint))
	model.load_variables_from_checkpoint(sess, model_load_checkpoint)

	# load input
	talk_id = 2772
	talk = Talk(talk_id)

	mfcc_width = int(2*training_data_provider.SEQUENCE_RADIUS / 0.005)
	mfcc_features = np.load(talk.features_path())
	interval_count = int(mfcc_features.shape[0] // mfcc_width)
	mfcc_features = mfcc_features[:interval_count*mfcc_width]
	mfcc_features = mfcc_features.reshape((interval_count,mfcc_width,13))

	# perform prediction
	prediction_vals = np.zeros((0,8*training_data_provider.word_embedding.EMBEDDING_DIMENSION))
	batch_size = 50
	while mfcc_features.shape[0] > 0:
		if batch_size > mfcc_features.shape[0]:
			batch_size = mfcc_features.shape[0]
		chunk = mfcc_features[:batch_size]
		mfcc_features = mfcc_features[batch_size:]
		val_prediction = sess.run(
			[prediction],
			feed_dict={
				input_3d: chunk
			}
		)
		prediction_vals = np.concatenate((prediction_vals, np.array(val_prediction).reshape((batch_size, 8*training_data_provider.word_embedding.EMBEDDING_DIMENSION))), axis=0)

	# release gpu resources
	sess.close()

	np.save("predictions.npy", prediction_vals)


