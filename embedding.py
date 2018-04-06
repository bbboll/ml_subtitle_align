import json
import os.path
import numpy as np
import absolute_path

EMBEDDING_DIMENSION = 25
BATCH_SIZE = int(1e4)

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
	"""
	Compute a stochastic approximation to the cost function value.
	"""
	cost = 0
	regularization = 0
	for i in np.random.randint(0, n**2-n, size=BATCH_SIZE): # range(n**2 - n):
		x_ind, y_ind = enumerate_square(i, n)
		emb_sim = cosine_sim(
					x_flat.reshape((n, EMBEDDING_DIMENSION))[x_ind],
					x_flat.reshape((n, EMBEDDING_DIMENSION))[y_ind]
				)
		cost += (emb_sim - pairwise_jaccard[i])**2
		regularization += (np.linalg.norm(x_flat.reshape((n, EMBEDDING_DIMENSION))[x_ind])-1)**2
		regularization += (np.linalg.norm(x_flat.reshape((n, EMBEDDING_DIMENSION))[y_ind])-1)**2
	return cost/BATCH_SIZE + regularization/(2*BATCH_SIZE)

def embedding_cost_gradient(x_flat, pairwise_jaccard, n):
	"""
	Compute a stochastic approximation of the cost function gradient.
	"""
	grad = np.zeros((len(x_flat),))
	for i in np.random.randint(0, n**2-n, size=BATCH_SIZE):
		k, j = enumerate_square(i, n)
		xj = x_flat.reshape((n, EMBEDDING_DIMENSION))[j]
		xk = x_flat.reshape((n, EMBEDDING_DIMENSION))[k]
		comp_norm = np.linalg.norm(xk)
		norms = comp_norm*np.linalg.norm(xj)
		a = k*EMBEDDING_DIMENSION
		b = a + EMBEDDING_DIMENSION
		if norms == 0:
			grad[a:b] += np.ones(EMBEDDING_DIMENSION)
			continue
		grad[a:b] += 4*(xj.dot(xk)/norms - pairwise_jaccard[i])/norms * xj
		grad[a:b] += 2*(1-1/comp_norm) * xk
	return grad / BATCH_SIZE


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

	# compute word embedding
	emb_path = absolute_path._get_full_path("data", "training", "word_embedding_{}.npy".format(EMBEDDING_DIMENSION))
	if os.path.isfile(emb_path):
		emb = np.load(emb_path).reshape(n*EMBEDDING_DIMENSION,)
	else:
		emb = np.zeros(n*EMBEDDING_DIMENSION)

	# ITERATION_LIMIT = int(1e4)
	# rate = 0.01
	# for iter_ind in range(ITERATION_LIMIT):
	# 	emb -= rate*embedding_cost_gradient(emb, pairwise_jaccard, n)
	# 	if iter_ind % 100 == 0:
	# 		print("Iteration {}".format(iter_ind))
	print("Reached (stochastic) gradient norm {}".format(np.linalg.norm(embedding_cost_gradient(emb, pairwise_jaccard, n))))
	print("Achieved (stochastic) optimal value: {}".format(embedding_cost_function(emb, pairwise_jaccard, n)))
	np.save(emb_path, emb.reshape((n, EMBEDDING_DIMENSION)))