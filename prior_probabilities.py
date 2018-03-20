import json
import os.path
import nltk

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

outpath = _path("data/training/word_priors.json")
if os.path.isfile(outpath):
	print("Already computed!")
	exit()

word_timings_path = _path("data/training/word_timings.json")
if not os.path.isfile(word_timings_path):
	print("Please compute word timings first.")
	exit()

frequent_words_path = _path("data/training/frequent_words.json")
if not os.path.isfile(frequent_words_path):
	print("Please compute frequent words first.")
	exit()

word_timings = json.load(open(word_timings_path))
frequent_words = json.load(open(frequent_words_path))
ps = nltk.stem.PorterStemmer()

words = {w: 0 for w in frequent_words}

total_sum = 0
for _, timings in word_timings.items():
	for (w,_) in timings:
		stem = ps.stem(w)
		if stem in frequent_words:
			words[stem] += 1
			total_sum += 1

words = {w: (count/total_sum) for w, count in words.items()}
with open(outpath, 'w', encoding="utf-8") as f:
		json.dump(words, f)