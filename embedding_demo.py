import embedding
import json
from absolute_path import _get_full_path

shingles = json.load(open(_get_full_path("data", "training", "word_shingles.json")))
words = json.load(open(_get_full_path("data", "training", "frequent_full_words.json")))
emb = embedding.Embedding(45)

def demo(i):
	vec = emb.embedding[i]
	for (j, dist) in emb.get_closest_words(vec):
		print("{} - {} - {}".format(words[j], dist, embedding.jaccard(shingles[i][0], shingles[j][0])))

if __name__ == '__main__':
	for i in [0,1,6,22,70,351,4129]:
		demo(i)
		print("")