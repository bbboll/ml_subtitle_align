import sys
import os
import operator
import json
import numpy as np
from preprocessing.talk import AllTalks
import nltk
import scipy.stats

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

INTERVAL_SIZE = 0.4
DATA_SD = 0.8

frequent_words_path = _path("data/training/frequent_words.json")
all_words_path = _path("data/talks/counts.json")
word_timings_path = _path("data/training/word_timings.json")

try:
	nltk.data.find('tokenizers/punkt')
except LookupError:
	nltk.download('punkt')
ps = nltk.stem.PorterStemmer()

def compute_most_frequent_words():
	if not os.path.isfile(all_words_path):
		print("Please run data_stats first.")
		exit()
	filter_words = [".", ",", ";", "-", "!", "?", "--", "(Laughter)", "Laughter"]
	frequent_words = json.load(open(all_words_path))
	frequent_words = {k: v for k, v in frequent_words.items() if not k in filter_words and v > 1}
	frequent_words = sorted(frequent_words.items(), key=operator.itemgetter(1), reverse=True)
	with open(frequent_words_path, 'w', encoding="utf-8") as f:
		json.dump([w for (w, _) in frequent_words[:1500]], f)

def compute_word_timings():
	# load or compute list of frequent words
	if not os.path.isfile(frequent_words_path):
		compute_most_frequent_words()
	frequent_words = json.load(open(frequent_words_path))

	# flush training data to file for each talk
	word_timings = {}
	for talk in AllTalks():
		try:
			print("Extracting word timings from talk {}: {}".format(talk.ID, talk.title))
			talk.load_subtitle()
			word_timings[talk.ID] = [(w, t) for (t, w) in talk.subtitle.words_with_timing if ps.stem(w) in frequent_words]
		except:
			pass
	with open(word_timings_path, "w", encoding="utf-8") as f:
		json.dump(word_timings, f)

def compute_word_probability(i, distrib):
	return distrib.cdf((i+1)*INTERVAL_SIZE)-distrib.cdf(i*INTERVAL_SIZE)


if __name__ == '__main__':
	# load or generate word timings
	if not os.path.isfile(word_timings_path):
		compute_word_timings()
	frequent_words = json.load(open(frequent_words_path))
	word_timings = json.load(open(word_timings_path))

	mfcc_per_interval = int(INTERVAL_SIZE / 0.005)
	features = np.zeros((0,mfcc_per_interval,13))
	labels = np.zeros((0,1500))

	for talk in AllTalks(limit=5):
		mfcc_features = np.load(talk.features_path())
		interval_count = int(mfcc_features.shape[0] // mfcc_per_interval)
		mfcc_features = mfcc_features[:interval_count*mfcc_per_interval]
		mfcc_features = mfcc_features.reshape((interval_count,mfcc_per_interval,13))
		features = np.concatenate((features, mfcc_features), axis=0)

		talk_labels = np.zeros((1500, interval_count))
		for (w, t) in word_timings[str(talk.ID)]:
			distrib = scipy.stats.norm(t, DATA_SD)
			word_ind = None
			try:
				word_ind = frequent_words.index(ps.stem(w))
			except ValueError:
				print("Could not find word {}".format(w))
				continue
			interval_ind = int(t // INTERVAL_SIZE)
			index_range = [interval_ind+i for i in range(-2,2) if interval_ind+i >= 0 and interval_ind+i+1 < interval_count]
			talk_labels[word_ind][index_range] += np.array([compute_word_probability(i, distrib) for i in index_range])
		labels = np.concatenate((labels, talk_labels.T), axis=0)
	print(features.shape)
	print(labels.shape)
