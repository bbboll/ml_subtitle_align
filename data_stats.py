import os.path
import json
import nltk
import numpy as np
from preprocessing.talk import AllTalks

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

def explore_counts():
	"""
	Explore word count data saved in data/talks/counts.json
	Result:
		The dataset contains a total of 5287125 (non-distinct) words.
		32210 words appear at least 2 times.
		They cover 0.99 of the text.
		23592 words appear at least 3 times.
		They cover 0.99 of the text.
		4498 words appear at least 50 times.
		They cover 0.95 of the text.
		2364 words appear at least 150 times.
		They cover 0.92 of the text.
		1956 words appear at least 200 times.
		They cover 0.90 of the text.
		1688 words appear at least 250 times.
		They cover 0.89 of the text.
		1488 words appear at least 300 times.
		They cover 0.88 of the text.
		1330 words appear at least 350 times.
		They cover 0.87 of the text.
		1200 words appear at least 400 times.
		They cover 0.86 of the text.
	"""
	data = json.load(open(_path("data/talks/counts.json")))
	filter_words = [".", ",", ";", "-", "!", "?", "--", "(Laughter)", "Laughter"]
	frequent_words = {k: v for k, v in data.items() if not k in filter_words}
	total_word_count = np.sum([v for k, v in data.items() if not k in filter_words])
	print("The dataset contains a total of {} (non-distinct) words.".format(total_word_count))

	for count in [1, 2, 49, 149, 199, 249, 299, 349, 399]:
		frequent_words = {k: v for k, v in frequent_words.items() if v > count}
		fraction = np.sum([v for _, v in frequent_words.items()]) / total_word_count
		print("{} words appear at least {} times.".format(len(frequent_words), count+1))
		print("They cover {0:.2f} of the text.".format(fraction))

nltk.download('punkt')

total_duration = 0
words = {}

ps = nltk.stem.PorterStemmer()

def count_words(text):
	"""
	Stem all words in the given string and increment their counts
	in the word dictionary
	"""
	text_words = nltk.word_tokenize(text)
	for w in text_words:
		stem = ps.stem(w)
		if not stem in words:
			words[stem] = 1
		else:
			words[stem] += 1

if __name__ == '__main__':
	for t in AllTalks():
		total_duration += t.duration / 60
		count_words(t.transcript)

	# save word stem counts to file
	with open(_path("data/talks/counts.json"), "w", encoding="utf-8") as f:
		json.dump(words, f)
	
	print("Counted {} distinct words in all transcripts combined.".format(len(words)))
	print("The total duration of all talks is {0:.2f}h".format(total_duration/60)) # 561.54h