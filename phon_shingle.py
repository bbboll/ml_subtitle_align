import json
import os.path

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

if __name__ == '__main__':
	"""
	Transcribe all frequent words in the dataset into a phonetic representation. Compute shingles from the phones and save them
	to a file.
	"""
	if not os.path.isfile(_path("data/training/frequent_full_word_phonetics.json")) or not os.path.isfile(_path("data/training/frequent_full_words.json")):
		print("Please execute data_phonetic_stats.py first.")
		exit()
	data = json.load(open(_path("data/training/frequent_full_word_phonetics.json")))
	words = json.load(open(_path("data/training/frequent_full_words.json")))
	word_shingles = []
	all_shingles = []
	for word_ind, word_phon in enumerate(data):
		shingle_variants = []
		for variant in word_phon:
			shingles = []
			if len(variant) == 1:
				if not variant in all_shingles:
					all_shingles.append(variant)
				shingle_variants.append([all_shingles.index(variant)])
			else:
				variant_shingles = [list(x) for x in zip(variant[:-1], variant[1:])]
				for shingle in variant_shingles:
					if not shingle in all_shingles:
						all_shingles.append(shingle)
				shingle_variants.append([all_shingles.index(s) for s in variant_shingles])
		word_shingles.append(shingle_variants)

	with open(_path("data/training/all_shingles.json"), "w", encoding="utf-8") as f:
		json.dump(all_shingles, f)
	with open(_path("data/training/word_shingles.json"), "w", encoding="utf-8") as f:
		json.dump(word_shingles, f)