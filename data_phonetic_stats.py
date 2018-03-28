import nltk
import json
import numpy as np
import training_routines
from preprocessing.talk import AllTalks
import extract_training_data as extractor

try:
	nltk.data.find('corpora/cmudict')
except LookupError:
	nltk.download('cmudict')

if __name__ == '__main__':
	
	frequent_words = json.load(open(training_routines._get_full_path("data", "training", "frequent_words.json")))
	full_words = {}
	for t in AllTalks():
		for word in nltk.word_tokenize(t.transcript):
			if extractor.ps.stem(word) in frequent_words:
				if word in full_words:
					full_words[word] += 1
				else:
					full_words[word] = 0
	print("The frequent stems account for {} distinct words with an accumulated count of {}".format(len(full_words), np.sum([c for _, c in full_words.items()])))
	# The frequent stems account for 8230 distinct words with an accumulated count of 4592945
	# This is about 86% of the whole dataset
	with open(training_routines._get_full_path("data", "training", "frequent_full_words.json"), "w", encoding="utf-8") as f:
		json.dump([w for w, c in full_words.items()], f)
