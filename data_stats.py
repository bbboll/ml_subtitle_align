import os.path
import json
import nltk
import numpy as np

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
		1488 words appear at least 250 times.
		They cover 0.88 of the text.
		1488 words appear at least 300 times.
		They cover 0.88 of the text.
	"""
	data = json.load(open("data/talks/counts.json"))
	filter_words = [".", ",", ";", "-", "!", "?", "--", "(Laughter)", "Laughter"]
	frequent_words = {k: v for k, v in data.items() if not k in filter_words}
	total_word_count = np.sum([v for k, v in data.items() if not k in filter_words])
	print("The dataset contains a total of {} (non-distinct) words.".format(total_word_count))
	
	for count in [1, 2, 49, 149, 199, 249, 299, 249, 299]:
		frequent_words = {k: v for k, v in frequent_words.items() if v > count}
		fraction = np.sum([v for _, v in frequent_words.items()]) / total_word_count
		print("{} words appear at least {} times.".format(len(frequent_words), count+1))
		print("They cover {0:.2f} of the text.".format(fraction))

nltk.download('punkt')

ids = []
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
	# iterate over talk JSON files
	chunk_size = 20
	for chunk_start in range(0, 2560, chunk_size):
		filename = "data/talks/ted_talks_{}.json".format(chunk_start)
		if not os.path.isfile(filename):
			break

		# iterate over all talks in the current file
		data = json.load(open(filename))
		for talk in data:
			if not talk["id"] in ids:
				ids.append(talk["id"])
				total_duration += int(talk["duration"]) / 60
				count_words(talk["transcript"])

	# save word stem counts to file
	with open("data/talks/counts.json", "w", encoding="utf-8") as f:
		json.dump(words, f)
	print("Counted {} distinct words in all transcripts combined.".format(len(words)))

	print("The total duration of all talks is {0:.2f}h".format(total_duration/60))
	# 561.54h