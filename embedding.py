import json
import os.path

def _path(relpath):
	"""
	Returns an absolute path for the given path (which is relative to the root directory ml_subtitle_align)
	"""
	current_dir = os.path.dirname(__file__)
	return os.path.abspath(os.path.join(current_dir, relpath))

def jaccard(x, y):
	"""
	Compute the jaccard distance of x and y.
	for sets x and y, this is the ratio of their intersection cardinality
	to their union cardinality
	jacc(x,y) = #(x n y) / #(x u y)
	This gives a value between 0 (totally different) and 1 (completely equal)
	"""
	intersection = []
	union = []
	for xi in x:
		if xi in y and not xi in intersection:
			intersection.append(xi)
		if not xi in union:
			union.append(xi)
	for yi in y:
		if not yi in union:
			union.append(yi)
	return len(intersection) / len(union)

if __name__ == '__main__':
	"""
	Compute a word embedding such that two embeddings have small cosine distance iff
	the respective words are similar sounding (Jaccard distance of their phoneme shingles)
	"""

	# TODO
	
	pass