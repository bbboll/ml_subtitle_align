import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from preprocessing.talk import Talk
import training_routines
import json
import os.path
import extract_training_data as extractor

arguments = argparse.ArgumentParser()
arguments.add_argument("id", help="TED talk id.", type=int)
arguments.add_argument("run", help="The training run from which to load the model (Path relative to ml_subtitle_align/training_data). This path needs to contain a training_config.json and a train/ directory with one or more checkpoints.", type=str)
options = arguments.parse_args()
talk_id = options.id

# load data preprocessing results
if not os.path.isfile(extractor.frequent_words_path) or not os.path.isfile(extractor.word_timings_path):
	print("Execute extract_training_data first.")
	exit()
frequent_words = json.load(open(extractor.frequent_words_path))
word_timings = json.load(open(extractor.word_timings_path))

# start a new tensorflow session
sess = tf.InteractiveSession()
input_3d = tf.placeholder(tf.float32, [None, 80, 13], name="input_3d")

# load model
model_load_checkpoint, training_config = training_routines.get_model_from_run(options.run)
model = training_routines.get_model_obj_from_config(training_config)
prediction = model.test_model(input_3d)

print("Loading model from checkpoint {}".format(model_load_checkpoint))
model.load_variables_from_checkpoint(sess, model_load_checkpoint)

# load input
talk = Talk(talk_id)

mfcc_per_interval = int(extractor.INTERVAL_SIZE / 0.005)
mfcc_features = np.load(talk.features_path())
interval_count = int(mfcc_features.shape[0] // mfcc_per_interval)
mfcc_features = mfcc_features[:interval_count*mfcc_per_interval]
mfcc_features = mfcc_features.reshape((interval_count,mfcc_per_interval,13))

# perform prediction
prediction_vals = np.zeros((0,1500))
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
	prediction_vals = np.concatenate((prediction_vals, np.array(val_prediction).reshape((batch_size, 1500))), axis=0)

# release gpu resources
sess.close()

true_labels = training_routines.compute_full_vector_labels(talk, interval_count)

#y = np.sum(prediction_vals, axis=1)
word_ind = frequent_words.index("we")
y = prediction_vals[:,word_ind] # word: "they"
print(y)
x = np.linspace(0,interval_count*0.4,num=interval_count)
markers = [ind for ind, val in enumerate(true_labels[:,word_ind]) if val > 0]
plt.plot(x, y, '-bD', markevery=markers)
plt.show()