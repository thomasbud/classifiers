from data import Dataset, Labels
from utils import evaluate
import math
import os, sys


class Rocchio:
	def __init__(self):
		# centroids vectors for each Label in the training set.
		self.centroids = {l: {} for l in Labels}
		# total number of documents for each label/class in the trainin set.
		self.n_doc = {l: 0 for l in Labels}


	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Loop over all the samples in the training set, convert the
		documents to vectors and find the centroid for each Label.
		"""
		doc_vectors = []
		for doc in ds:
			dic = {}
			self.n_doc[doc[2]] += 1
			for word in doc[1].split():					
				if word.lower() in dic:
					dic[word.lower()] += 1
				else:
					dic[word.lower()] = 1
			tup = (doc[2], dic)
			doc_vectors.append(tup)

		for doc in doc_vectors:
			magnitude = 0
			for word in doc[1]:
				magnitude += doc[1][word]**2
			magnitude = magnitude**0.5

			for word in doc[1]:
				doc[1][word] = doc[1][word] / magnitude

		for doc in doc_vectors:
			for word in doc[1]:
				if word in self.centroids[doc[0]]:
					self.centroids[doc[0]][word] += doc[1][word]
				else:
					self.centroids[doc[0]][word] = doc[1][word]

		for label in self.centroids:
			for word in self.centroids[label]:
				self.centroids[label][word] = self.centroids[label][word] / self.n_doc[label]
		

	def predict(self, x):
		"""
		x: string of words in the document.
		
		TODO: Convert x to vector, find the closest centroid and return the
		label corresponding to the closest centroid.
		"""
		#collect raw term freqs
		A = {}
		for word in x.split():
			if word.lower() in A:
				A[word.lower()] += 1
			else:
				A[word.lower()] = 1

		#normalize
		magnitude = 0
		for word in A:
			magnitude += A[word]**2
		magnitude = magnitude**0.5

		for word in A:
			A[word] = A[word] / magnitude

		#cosine similarity
		score = []
		A_magnitude = 0
		for value in A:
			A_magnitude += A[value]**2

		for B in self.centroids:
			numerator = 0
			B_magnitude = 0
			for word in A:
				if word in self.centroids[B]:
					numerator += A[word] * self.centroids[B][word]
			for value in self.centroids[B]:
				B_magnitude += self.centroids[B][value]**2
			denominator = A_magnitude**0.5 * B_magnitude**0.5
			tup = (B, numerator/denominator)
			score.append(tup)
		
		#assign class label
		def getScore(item):
			return item[1]
		
		sorted_score = sorted(score, key=getScore, reverse=True)
		return sorted_score[0][0]

def main(train_split):
	rocchio = Rocchio()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	rocchio.train(ds)

	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(rocchio, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(rocchio, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(rocchio, test_ds)

if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
