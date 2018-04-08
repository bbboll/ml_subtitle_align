import json
import os.path
import numpy as np
import absolute_path

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

def embedding_cost_function(x_flat, pairwise_jaccard, n, dim):
	"""
	Compute a stochastic approximation to the cost function value.
	"""
	cost = 0
	regularization = 0
	for i in np.random.randint(0, n**2-n, size=BATCH_SIZE): # range(n**2 - n):
		x_ind, y_ind = enumerate_square(i, n)
		emb_sim = cosine_sim(
					x_flat.reshape((n, dim))[x_ind],
					x_flat.reshape((n, dim))[y_ind]
				)
		cost += (emb_sim - pairwise_jaccard[i])**2
		#regularization += (np.linalg.norm(x_flat.reshape((n, dim))[x_ind])-1)**2
		#regularization += (np.linalg.norm(x_flat.reshape((n, dim))[y_ind])-1)**2
	return cost/BATCH_SIZE + regularization/(2*BATCH_SIZE)

def embedding_cost_gradient(x_flat, pairwise_jaccard, n, dim):
	"""
	Compute a stochastic approximation of the cost function gradient.
	"""
	grad = np.zeros((len(x_flat),))
	for i in np.random.randint(0, n**2-n, size=BATCH_SIZE):
		k, j = enumerate_square(i, n)
		xj = x_flat.reshape((n, dim))[j]
		xk = x_flat.reshape((n, dim))[k]
		comp_norm = np.linalg.norm(xk)
		norms = comp_norm*np.linalg.norm(xj)
		a = k*dim
		b = a + dim
		if norms == 0:
			grad[a:b] += np.random.rand(dim)*2-1
			continue
		grad[a:b] += 2*(cosine_sim(xj, xk) - pairwise_jaccard[i])/norms * xj
		#grad[a:b] += 2*(1-1/comp_norm) * xk
	return grad / BATCH_SIZE

class Embedding(object):
	"""
	Compute a word embedding such that two embeddings have small cosine distance iff
	the respective words are similar sounding (Jaccard distance of their phoneme shingles)
	We normalize the embeddings to have norm 1 (by regularization). The norm can be used as
	a measure for how likely the classified region is to contain any word at all. (0 as None label)
	"""
	def __init__(self, dim):
		super(Embedding, self).__init__()
		self.EMBEDDING_DIMENSION = dim
		self.word_shingles = json.load(open(absolute_path._get_full_path("data", "training", "word_shingles.json")))
		self.n = len(self.word_shingles)
		self.load_pairwise_jaccard_distances()
		self.load_embedding()

	def load_pairwise_jaccard_distances(self):
		"""
		"""
		pairwise_jaccard_path = absolute_path._get_full_path("data", "training", "pairwise_jaccard.npy")
		if os.path.isfile(pairwise_jaccard_path):
			self.pairwise_jaccard = np.load(pairwise_jaccard_path)
		else:
			# compute pairwise jaccard distances
			self.pairwise_jaccard = []
			for i in range(self.n**2 - self.n):
				row, col = enumerate_square(i, self.n)
				jacc_sim = 0
				for ind_x, x in enumerate(self.word_shingles[row]):
					for y in self.word_shingles[col][ind_x:]:
						jacc_sim = max(jacc_sim, jaccard(x, y))
				self.pairwise_jaccard.append(jacc_sim)
			np.save(pairwise_jaccard_path, np.array(self.pairwise_jaccard))


	def load_embedding(self):
		"""
		Load word embedding for given dimension.
		"""
		emb_path = absolute_path._get_full_path("data", "training", "word_embedding_{}.npy".format(self.EMBEDDING_DIMENSION))
		if os.path.isfile(emb_path):
			self.embedding = np.load(emb_path)
			return
		else:
			self.embedding = (np.random.rand(self.n*self.EMBEDDING_DIMENSION)*2 - 1).reshape((self.n, self.EMBEDDING_DIMENSION))
	
	def optimize_embedding(self, step_limit = int(3e4)):
		"""
		"""
		rate = 0.01
		self.embedding = self.embedding.reshape((self.n*self.EMBEDDING_DIMENSION,))
		for iter_ind in range(step_limit):
			self.embedding -= rate*embedding_cost_gradient(self.embedding, self.pairwise_jaccard, self.n, self.EMBEDDING_DIMENSION)
			if iter_ind % 100 == 0:
				print("Iteration {}".format(iter_ind))
		print("Reached (stochastic) gradient norm {}".format(np.linalg.norm(embedding_cost_gradient(self.embedding, self.pairwise_jaccard, self.n, self.EMBEDDING_DIMENSION))))
		print("Achieved (stochastic) optimal value: {}".format(embedding_cost_function(self.embedding, self.pairwise_jaccard, self.n, self.EMBEDDING_DIMENSION)))
		self.embedding = self.embedding.reshape((self.n, self.EMBEDDING_DIMENSION))
		emb_path = absolute_path._get_full_path("data", "training", "word_embedding_{}.npy".format(self.EMBEDDING_DIMENSION))
		np.save(emb_path, self.embedding)


if __name__ == '__main__':
	"""
	"""
	emb = Embedding(15)
	emb.optimize_embedding(step_limit=int(1e4))