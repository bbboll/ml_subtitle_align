import json
import os.path
import numpy as np
import absolute_path
from scipy.optimize import fmin_bfgs

EMBEDDING_DIMENSION = 45

def cosine_sim(x, y, normalize=True):
	"""
	Compute the cosine similarity of x and y.
	For vectors x and y, this is the cosine of the angle between them.
	This gives a value between -1 (totally opposite) and 1 (completely equal direction)
	If normalize is set to True, the result is linearly transformed into [0,1] for better
	comparability to the jaccard similarity.
	"""
	norms = np.linalg.norm(x)*np.linalg.norm(y)
	if norms == 0:
		return 0
	cosine = np.dot(x,y) / norms
	if normalize:
		return 0.5*cosine + 0.5
	return cosine

def jaccard(x, y):
	"""
	Compute the jaccard similarity of x and y.
	For sets x and y, this is the ratio of their intersection cardinality
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

def enumerate_square(i, n):
	"""
	Given i in the range(n^2-n) compute a bijective mapping
	range(n^2-n) -> range(n)*range(n-1) 
	"""
	row = int(i // (n-1))
	col = int(i % (n-1))
	if col >= row:
		col += 1
	return row, col

def embedding_cost_function(x_flat, pairwise_jaccard, n):
	cost = 0
	for i in range(n**2 - n):
		x_ind, y_ind = enumerate_square(i, n)
		emb_sim = cosine_sim(
					x_flat.reshape((n, EMBEDDING_DIMENSION))[x_ind],
					x_flat.reshape((n, EMBEDDING_DIMENSION))[y_ind]
				)
		cost += (emb_sim - pairwise_jaccard[i])**2
	return cost + (np.linalg.norm(x_flat)-1)**2

if __name__ == '__main__':
	"""
	Compute a word embedding such that two embeddings have small cosine distance iff
	the respective words are similar sounding (Jaccard distance of their phoneme shingles)
	We normalize the embeddings to have norm 1 (by regularization). The norm can be used as
	a measure for how likely the classified region is to contain any word at all. (0 as None label)
	"""
	word_shingles = json.load(open(absolute_path._get_full_path("data", "training", "word_shingles.json")))
	n = len(word_shingles)

	# compute pairwise jaccard distances
	pairwise_jaccard_path = absolute_path._get_full_path("data", "training", "pairwise_jaccard.npy")
	if os.path.isfile(pairwise_jaccard_path):
		pairwise_jaccard = np.load(pairwise_jaccard_path)
	else:
		pairwise_jaccard = []
		for i in range(n**2 - n):
			row, col = enumerate_square(i, n)
			jacc_sim = 0
			for ind_x, x in enumerate(word_shingles[row]):
				for y in word_shingles[col][ind_x:]:
					jacc_sim = max(jacc_sim, jaccard(x, y))
			pairwise_jaccard.append(jacc_sim)
		np.save(pairwise_jaccard_path, np.array(pairwise_jaccard))

	print("start")
	print(embedding_cost_function(np.ones(n*EMBEDDING_DIMENSION), pairwise_jaccard, n))
	exit()

	# compute word embedding
	emb = np.zeros(n*EMBEDDING_DIMENSION)
	emb, fopt = fmin_bfgs(
		embedding_cost_function,
		emb,
		args = (pairwise_jaccard, n),
		maxiter = 1
	)
	print("Achieved optimal value: {}".format(fopt))