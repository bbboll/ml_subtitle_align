import numpy as np
import json
import extract_training_data as extractor
import os.path
from preprocessing.talk import AllTalks
from sklearn.model_selection import train_test_split
import scipy.stats
from absolute_path import _get_full_path
from embedding import Embedding

if not os.path.isfile(extractor.frequent_words_path) or not os.path.isfile(extractor.word_timings_path):
	print("Execute extract_training_data first.")
	exit()
frequent_words = json.load(open(extractor.frequent_words_path))
word_timings = json.load(open(extractor.word_timings_path))
mfcc_per_interval = int(extractor.INTERVAL_SIZE / 0.005)
embedded_words = json.load(open(_get_full_path("data", "training", "frequent_full_words.json")))

word_embedding = Embedding(15)
SEQUENCE_RADIUS = 1

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

def compute_interval_sequence_data(talk, mfcc_features):
	"""
	Look for suited 2-shingles of embedded words in the talk transcript
	and batch them together with a larger interval of mfcc features
	"""
	feature_count = int(2*SEQUENCE_RADIUS // 0.005)
	features = np.zeros((0, feature_count*13))
	labels = np.zeros((0, 2*word_embedding.EMBEDDING_DIMENSION))
	for ((w1, t1), (w2, t2)) in zip(word_timings[str(talk.ID)][:-1], word_timings[str(talk.ID)][1:]):
		if w1 in embedded_words and w2 in embedded_words:
			if t2-t1 > 1:
				# ignore word pairs which are too close to each other
				continue
			mean_time = (t1+t2)/2
			mfcc_start_time = mean_time - SEQUENCE_RADIUS
			mfcc_start_index = int(mfcc_start_time // 0.005)
			mfcc_end_index = mfcc_start_index  + feature_count
			if mfcc_features.shape[0] < mfcc_end_index + 1:
				# skip pair, if it is too close the the audio end
				continue
			features = np.concatenate((features, mfcc_features[mfcc_start_index:mfcc_end_index, :].reshape((1, feature_count*13))), axis=0)
			labels = np.concatenate((labels, np.concatenate((word_embedding.embed(w1), word_embedding.embed(w2))).reshape((1, 2*word_embedding.EMBEDDING_DIMENSION))), axis=0)
	return features.reshape((features.shape[0], feature_count, 13)), labels


class DataProvider(object):
	"""
	Provides training data batches (input + labels)
	for a given training config. 
	"""
	def __init__(self, training_config):
		self.training_config = training_config
		self.batch_size = training_config["batch_size"]
		self.data_model =training_config["data"]
		self.full = (not training_config["loss"]["function"] == "softmax_cross_entropy")

	def get_feature_count(self):
		if self.data_model == "interval_sequence":
			return int(2*SEQUENCE_RADIUS // 0.005)
		return 80

	def xbatches(self, training=True):
		"""Batch MFCC data, labels together.
		""" 
		talk_limit = None
		all_ids = [talk.ID for talk in AllTalks(limit=talk_limit)]
		train_ids, test_ids = train_test_split(all_ids, test_size=0.1, shuffle=False)
		if self.data_model == "interval_sequence":
			features = np.zeros((0, int(2*SEQUENCE_RADIUS // 0.005), 13))
			labels = np.zeros((0, 2*word_embedding.EMBEDDING_DIMENSION))
		else:
			features = np.zeros((0,mfcc_per_interval,13))
			labels = np.zeros((0,1500))
		talks = AllTalks(limit=talk_limit)

		try:
			while True:
				if labels.shape[0] < self.batch_size:
					talk = next(talks)
					if training and not talk.ID in train_ids:
						continue
					if not training and talk.ID in train_ids:
						continue
					if not str(talk.ID) in word_timings:
						continue

					# load all talk features
					mfcc_features = np.load(talk.features_path())

					if self.data_model == "interval_sequence":
						# 
						talk_features, talk_labels = compute_interval_sequence_data(talk, mfcc_features)
						features = np.concatenate((features, talk_features), axis=0)
						labels = np.concatenate((labels, talk_labels), axis=0)
						
					else:
						# append mfcc features to all features
						interval_count = int(mfcc_features.shape[0] // mfcc_per_interval)
						mfcc_features = mfcc_features[:interval_count*mfcc_per_interval,:]
						mfcc_features = mfcc_features.reshape((interval_count,mfcc_per_interval,13))
						features = np.concatenate((features, mfcc_features), axis=0)

						if self.full:
							talk_labels = compute_full_vector_labels(talk, interval_count)
						else:
							talk_labels = compute_multi_categorical_labels(talk, interval_count)
						labels = np.concatenate((labels, talk_labels), axis=0)
				else:
					yield (
						features[:self.batch_size],
						labels[:self.batch_size]
					)
					features = features[self.batch_size:]
					labels = labels[self.batch_size:]
		except StopIteration:
			return
