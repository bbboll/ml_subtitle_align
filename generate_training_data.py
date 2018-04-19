import training_data_provider
import numpy as np
from absolute_path import _get_full_path
import training_routines
import os.path
import json

if __name__ == '__main__':
	# load config
	config_path = training_routines._get_full_path("training_config.json")
	if not os.path.isfile(config_path):
		copyfile(config_path+".default", config_path)
	config = json.load(open(config_path))
	
	data_provider = training_data_provider.DataProvider(config, categorical_labels=True)

	SAVE_INTERVAL = 100
	savepath = _get_full_path("data", "training", "categorical")

	# populate training data
	training_features = np.zeros((0, int(2*training_data_provider.SEQUENCE_RADIUS // 0.005), 13))
	training_labels = np.zeros((0, 2))
	for i, (train_input, train_ground_truth) in enumerate(data_provider.xbatches(training=True)):
		print("batch", i)
		training_features = np.concatenate((training_features, train_input), axis=0)
		training_labels = np.concatenate((training_labels, train_ground_truth), axis=0)

		if i > 0 and i % SAVE_INTERVAL == 0:
			print("saving...")
			np.save(os.path.join(savepath, "training_features_{}.npy".format(i)), training_features)
			np.save(os.path.join(savepath, "training_labels_{}.npy".format(i)), training_labels)
			training_features = np.zeros((0, int(2*training_data_provider.SEQUENCE_RADIUS // 0.005), 13))
			training_labels = np.zeros((0, 2))
			print("done")
	print("saving...")
	np.save(os.path.join(savepath, "training_features_{}.npy".format(i)), training_features)
	np.save(os.path.join(savepath, "training_labels_{}.npy".format(i)), training_labels)
	print("done")

	# populate test data
	testing_features = np.zeros((0, int(2*training_data_provider.SEQUENCE_RADIUS // 0.005), 13))
	testing_labels = np.zeros((0, 2))
	for i, (test_input, test_ground_truth) in enumerate(data_provider.xbatches(training=False)):
		testing_features = np.concatenate((testing_features, test_input), axis=0)
		testing_labels = np.concatenate((testing_labels, test_ground_truth), axis=0)

		if i > 0 and i % SAVE_INTERVAL == 0:
			print("saving...")
			np.save(os.path.join(savepath, "test_features_{}.npy".format(i)), testing_features)
			np.save(os.path.join(savepath, "test_labels_{}.npy".format(i)), testing_labels)
			testing_features = np.zeros((0, int(2*training_data_provider.SEQUENCE_RADIUS // 0.005), 13))
			testing_labels = np.zeros((0, 2))
			print("done")
	print("saving...")
	np.save(os.path.join(savepath, "test_features_{}.npy".format(i)), testing_features)
	np.save(os.path.join(savepath, "test_labels_{}.npy".format(i)), testing_labels)
	print("done")

