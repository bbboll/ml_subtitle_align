import numpy as np
import os
import datetime
import hashlib
import json
import extract_training_data as extractor
from sklearn.model_selection import train_test_split
from preprocessing.talk import AllTalks
import scipy.stats
import re
import models.conv_model
import models.conv_lstm_model
import models.deep_conv_model
import models.big_deep_conv_model
import models.conv_deep_nn

def _get_full_path(*rel_path):
	"""Make absolute path to a file or directory in the project folder ml_subtitle_align.

	Arguments:
        *rel_path: List of path elements.

    Returns:
        `str`: Absolute path to requested file or directory.
	"""
	path = os.path.abspath(__file__) # `.../ml_subtitle_align/train.py`
	path = os.path.dirname(path) # `.../ml_subtitle_align/`
	return os.path.join(path, *rel_path)

def get_training_save_paths(model_config):
	config_hash = hashlib.md5("-".join(["{}:{}".format(k,v) for k,v in model_config.items()]).encode('utf-8')).hexdigest()
	dirpath = _get_full_path("training_data", "run_{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H"), config_hash))
	if not os.path.isdir(dirpath):
		os.mkdir(dirpath)
	return dirpath, os.path.join(dirpath, "retrain_logs"), os.path.join(dirpath, "train")



if not os.path.isfile(extractor.frequent_words_path) or not os.path.isfile(extractor.word_timings_path):
	print("Execute extract_training_data first.")
	exit()
frequent_words = json.load(open(extractor.frequent_words_path))
word_timings = json.load(open(extractor.word_timings_path))
mfcc_per_interval = int(extractor.INTERVAL_SIZE / 0.005)

def compute_full_vector_labels(talk, interval_count):
	"""
	For each interval, return a vector of shape (,1500) containing every
	probability for each single word.
	"""
	talk_labels = np.zeros((1500, interval_count))
	for (w, t) in word_timings[str(talk.ID)]:
		distrib = scipy.stats.norm(t, extractor.DATA_SD)
		word_ind = None
		try:
			word_ind = frequent_words.index(extractor.ps.stem(w))
		except ValueError:
			print("Could not find word {}".format(w))
			continue
		interval_ind = int(t // extractor.INTERVAL_SIZE)
		radius = 2
		index_range = [interval_ind+i for i in range(-radius,radius+1) if interval_ind+i >= 0 and interval_ind+i+1 < interval_count]
		talk_labels[word_ind][index_range] += np.array([extractor.compute_word_probability(i, distrib) for i in index_range])
	return talk_labels.T

def compute_multi_categorical_labels(talk, interval_count):
	"""
	For each interval, return a vector of shape (,1500) containing integer
	labels in (0,1).
	"""
	talk_labels = np.zeros((1500, interval_count), dtype=int)
	for (w, t) in word_timings[str(talk.ID)]:
		distrib = scipy.stats.norm(t, extractor.DATA_SD)
		word_ind = None
		try:
			word_ind = frequent_words.index(extractor.ps.stem(w))
		except ValueError:
			print("Could not find word {}".format(w))
			continue
		interval_ind = int(t // extractor.INTERVAL_SIZE)
		talk_labels[word_ind][interval_ind] = 1
	return talk_labels.T

def xbatches(batch_size, training=True, full=True):
	"""Batch MFCC data, labels together.
	"""
	talk_limit = None
	all_ids = [talk.ID for talk in AllTalks(limit=talk_limit)]
	train_ids, test_ids = train_test_split(all_ids, test_size=0.1, shuffle=False)
	features = np.zeros((0,mfcc_per_interval,13))
	labels = np.zeros((0,1500))
	talks = AllTalks(limit=talk_limit)

	try:
		while True:
			if labels.shape[0] < batch_size:
				talk = next(talks)
				if training and not talk.ID in train_ids:
					continue
				if not training and talk.ID in train_ids:
					continue
				if not str(talk.ID) in word_timings:
					continue
				mfcc_features = np.load(talk.features_path())
				interval_count = int(mfcc_features.shape[0] // mfcc_per_interval)
				mfcc_features = mfcc_features[:interval_count*mfcc_per_interval,:]
				mfcc_features = mfcc_features.reshape((interval_count,mfcc_per_interval,13))
				features = np.concatenate((features, mfcc_features), axis=0)

				if full:
					talk_labels = compute_full_vector_labels(talk, interval_count)
				else:
					talk_labels = compute_multi_categorical_labels(talk, interval_count)
				labels = np.concatenate((labels, talk_labels), axis=0)
			else:
				yield (
					features[:batch_size],
					labels[:batch_size]
				)
				features = features[batch_size:]
				labels = labels[batch_size:]
	except StopIteration:
		return

def get_model_from_run(run):
	"""
	Given a relative path of shape
		run_2018-03-21-23_9fbe4594184ad6a9224f865a2bdfd407/
	extract the (absolute) model checkpoint and training_config.json paths
	"""
	checkpoint_meta_path = _get_full_path("training_data", run, "train", "checkpoint")
	checkpoint_path = ""
	with open(checkpoint_meta_path, "r") as f:
		line = f.readline()
		abs_checkpoint_path = re.search(r'\"(.+)\"', line).group(0)
		checkpoint_num = re.search(r'model.ckpt-([0-9]+)', abs_checkpoint_path).group(0)
		checkpoint_path = _get_full_path("training_data", run, "train", checkpoint_num)
	train_config = json.load(open(_get_full_path("training_data", run, "training_config.json")))
	return checkpoint_path, train_config

def get_model_obj_from_config(training_config):
	if training_config["model"] == "simple_conv":
		return models.conv_model.Model()
	elif training_config["model"] == "dense_conv":
		return models.conv_model.Model(hyperparams=["dense"])
	elif training_config["model"] == "conv_lstm":
		return models.conv_lstm_model.Model()
	elif training_config["model"] == "deep_conv":
		return models.deep_conv_model.Model()
	elif training_config["model"] == "big_deep_conv":
		return models.big_deep_conv_model.Model()
	elif training_config["model"] == "conv_deep_nn":
		return models.conv_deep_nn.Model()
	else: # if training_config["model"] == "experimental"
		return models.model.Model()