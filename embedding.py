import json
import os.path
import numpy as np
import absolute_path

BATCH_SIZE = int(3e4)

def cosine_dist(x, y, normalize=True):
	"""
	Compute the cosine distance of x and y.
	For vectors x and y, this is the cosine of the angle between them.
	This gives a value between -1 (totally opposite) and 1 (completely equal direction)
	If normalize is set to True, the result is linearly transformed into [0,1] for better
	comparability to the jaccard distance.
	"""
	norms = np.linalg.norm(x)*np.linalg.norm(y)
	if norms == 0:
		return 0
	cosine = np.dot(x,y) / norms
	if normalize:
		return 1 - 0.5*cosine + 0.5
	return -cosine

def jaccard(x, y):
	"""
	Compute the jaccard distance of x and y.
	For sets x and y, this is the ratio of their intersection cardinality
	to their union cardinality
	jacc(x,y) = 1 - #(x n y) / #(x u y)
	This gives a value between 0 (completely equal) and 1 (totally different)
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
	return 1 - len(intersection) / len(union)

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
		emb_sim = np.linalg.norm(x_flat.reshape((n, dim))[x_ind]-x_flat.reshape((n, dim))[y_ind])
		# emb_sim = cosine_dist(
		# 			x_flat.reshape((n, dim))[x_ind],
		# 			x_flat.reshape((n, dim))[y_ind]
		# 		)
		cost += (emb_sim - pairwise_jaccard[i])**2
	return cost/BATCH_SIZE # + regularization/(2*BATCH_SIZE)

def embedding_cost_gradient(x_flat, pairwise_jaccard, pairwise_jaccard_ind, n, dim):
	"""
	Compute a stochastic approximation of the cost function gradient.
	"""
	grad = np.zeros((len(x_flat),))
	#for i in np.random.randint(0, n**2-n, size=BATCH_SIZE):
	ratio_count = int(0.1*len(pairwise_jaccard))
	for i in np.random.choice(pairwise_jaccard_ind, BATCH_SIZE):
		k, j = enumerate_square(i, n)
		xj = x_flat.reshape((n, dim))[j]
		xk = x_flat.reshape((n, dim))[k]
		#comp_norm = np.linalg.norm(xk)
		norms = np.linalg.norm(xk-xj)
		#norms = comp_norm*np.linalg.norm(xj)
		a = k*dim
		b = a + dim
		if norms == 0:
			grad[a:b] += np.random.random_sample(dim)*2-1
			continue
		grad[a:b] += 2*(norms - pairwise_jaccard[i])/norms * (xk-xj)
		#grad[a:b] += -2*(cosine_dist(xj, xk) - pairwise_jaccard[i])/norms * xj

	for i in np.random.choice(len(pairwise_jaccard), int(0.02*BATCH_SIZE)):
		k, j = enumerate_square(i, n)
		xj = x_flat.reshape((n, dim))[j]
		xk = x_flat.reshape((n, dim))[k]
		#comp_norm = np.linalg.norm(xk)
		norms = np.linalg.norm(xk-xj)
		#norms = comp_norm*np.linalg.norm(xj)
		a = k*dim
		b = a + dim
		if norms == 0:
			grad[a:b] += np.random.random_sample(dim)*2-1
			continue
		grad[a:b] += 2*(norms - pairwise_jaccard[i])/norms * (xk-xj)
		#grad[a:b] += -2*(cosine_dist(xj, xk) - pairwise_jaccard[i])/norms * xj
	return grad / (1.02*BATCH_SIZE)

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
		self.words = json.load(open(absolute_path._get_full_path("data", "training", "frequent_full_words.json")))
		self.n = len(self.word_shingles)
		self.load_pairwise_jaccard_distances()
		self.load_embedding()

	def embed(self, word):
		word_ind = self.words.index(word)
		return self.embedding[word_ind]

	def get_index(self, word):
		return self.words.index(word)

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
						jacc_sim = max(jacc_sim, 1-jaccard(x, y))
				self.pairwise_jaccard.append(jacc_sim)
			np.save(pairwise_jaccard_path, np.array(self.pairwise_jaccard))
		self.pairwise_jaccard = 1-self.pairwise_jaccard
		self.pairwise_jaccard[np.where(self.pairwise_jaccard >= 1)] += 2

	def load_embedding(self):
		"""
		Load word embedding for given dimension.
		"""
		emb_path = absolute_path._get_full_path("data", "training", "word_embedding_{}.npy".format(self.EMBEDDING_DIMENSION))
		if os.path.isfile(emb_path):
			self.embedding = np.load(emb_path)
			return
		else:
			self.embedding = 5*(np.random.random_sample(self.n*self.EMBEDDING_DIMENSION)*2 - 1).reshape((self.n, self.EMBEDDING_DIMENSION))
	
	def optimize_embedding(self, step_limit = int(3e4)):
		"""
		"""
		rate = 0.01
		self.embedding = self.embedding.reshape((self.n*self.EMBEDDING_DIMENSION,))
		pairwise_jaccard_ind = np.array(np.where(self.pairwise_jaccard < 1)).flatten()
		for iter_ind in range(step_limit):
			self.embedding -= rate*embedding_cost_gradient(self.embedding, self.pairwise_jaccard, pairwise_jaccard_ind, self.n, self.EMBEDDING_DIMENSION)
			if iter_ind % 100 == 0:
				print("Iteration {}".format(iter_ind))
		print("Reached (stochastic) gradient norm {}".format(np.linalg.norm(embedding_cost_gradient(self.embedding, self.pairwise_jaccard, pairwise_jaccard_ind, self.n, self.EMBEDDING_DIMENSION))))
		print("Achieved (stochastic) optimal value: {}".format(embedding_cost_function(self.embedding, self.pairwise_jaccard, self.n, self.EMBEDDING_DIMENSION)))
		self.embedding = self.embedding.reshape((self.n, self.EMBEDDING_DIMENSION))
		emb_path = absolute_path._get_full_path("data", "training", "word_embedding_{}.npy".format(self.EMBEDDING_DIMENSION))
		np.save(emb_path, self.embedding)

	def get_closest_words(self, vec, limit=10):
		"""
		"""
		#distances = [cosine_dist(x, vec) for x in self.embedding]
		distances = [np.linalg.norm(x - vec) for x in self.embedding]
		closest = []
		for _ in range(limit):
			i = np.argmin(distances)
			closest.append((i, distances[i]))
			distances[i] = np.inf
		return closest


if __name__ == '__main__':
	"""
	"""
	emb = Embedding(15)
	emb.optimize_embedding(step_limit=int(5e4))
