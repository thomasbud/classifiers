from data import Dataset, Labels
from utils import evaluate
import os, sys

K = 3

class KNN:
	def __init__(self):
		# bag of words document vectors
		self.bow = []

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Save all the documents in the train dataset (ds) in self.bow.
		You need to transform the documents into vector space before saving
		in self.bow.
		"""
		for doc in ds:
			dic = {}
			for word in doc[1].split():
				if word.lower() in dic:
					dic[word.lower()] += 1
				else:
					dic[word.lower()] = 1

			tup = (doc[2], dic)
			self.bow.append(tup)

	def predict(self, x):
		"""
		x: string of words in the document.

		TODO: Predict class for x.
		1. Transform x to vector space.
		2. Find k nearest neighbors.
		3. Return the class which is most common in the neighbors.
		"""
		import statistics
		from statistics import mode

		score = []
		A = {}
		for word in x.split():
			if word.lower() in A:
				A[word.lower()] += 1
			else:
				A[word.lower()] = 1
		
		A_magnitude = 0
		for value in A:
			A_magnitude += A[value]**2

		for B in self.bow:
			numerator = 0
			B_magnitude = 0
			for word in A:
				if word in B[1]:
					numerator += A[word] * B[1][word]
			for value in B[1]:
				B_magnitude += B[1][value]**2
			denominator = A_magnitude**0.5 * B_magnitude**0.5
			tup = (B[0], numerator/denominator)
			score.append(tup)

		def getScore(item):
			return item[1]
		
		sorted_score = sorted(score, key=getScore, reverse=True)
		top_scores = []
		for i in range(K):
			top_scores.append(sorted_score[i][0])
		
		return mode(top_scores)

def main(train_split):
	knn = KNN()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	knn.train(ds)

	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(knn, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(knn, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(knn, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
