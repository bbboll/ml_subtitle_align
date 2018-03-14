import sys
import os
import operator
import json
from preprocessing.talk import AllTalks
import nltk

nltk.download('punkt')

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

INTERVAL_SIZE = 0.4
frequent_words_path = _path("data/training/frequent_words.json")
all_words_path = _path("data/talks/counts.json")
word_timings_path = _path("data/training/word_timings.json")

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

	ps = nltk.stem.PorterStemmer()

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


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("No interval size was given. Default is {}sec".format(INTERVAL_SIZE))
	else:
		INTERVAL_SIZE = float(sys.argv[1])
	savepath = "data/training/{}".format(int(INTERVAL_SIZE*1000))
	if not os.path.isdir(_path(savepath)):
		os.mkdir(savepath)

	if not os.path.isfile(word_timings_path):
		compute_word_timings()
	

