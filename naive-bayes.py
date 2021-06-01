from data import Dataset, Labels
from utils import evaluate
import math
import os, sys


class NaiveBayes:
	def __init__(self):
		# total number of documents in the training set.
		self.n_doc_total = 0
		# total number of documents for each label/class in the trainin set.
		self.n_doc = {l: 0 for l in Labels}
		# frequency of words for each label in the trainng set.
		self.vocab = {l: {} for l in Labels}

		# value of v
		self.v = 0
		# word frequency in label
		self.freq = []

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Loop over the dataset (ds) and update self.n_doc_total,
		self.n_doc and self.vocab.
		"""
		v_counter = {}
		for doc in ds:
			self.n_doc_total += 1
			self.n_doc[doc[2]] += 1
			for word in doc[1].split():
				if word.lower() not in v_counter:
					v_counter[word.lower()] = 1
					
				if word.lower() in self.vocab[doc[2]]:
					self.vocab[doc[2]][word.lower()] += 1
				else:
					self.vocab[doc[2]][word.lower()] = 1

		self.v = len(v_counter)
		
		for label in Labels:
			count = sum(self.vocab[label].values())
			self.freq.append(count)
			for value in self.vocab[label]:
				self.vocab[label][value] = (self.vocab[label][value] + 1) / (count + self.v + 1)

	def predict(self, x):
		"""
		x: string of words in the document.
		
		TODO: Use self.n_doc_total, self.n_doc and self.vocab to calculate the
		prior and likelihood probabilities.
		Add the log of prior and likelihood probabilities.
		Use MAP estimation to return the Label with hight score as
		the predicted label.
		"""
		CNB = []
		counter = -1
		for label in Labels:
			sum = 0
			counter += 1
			for word in x.split():
				if word.lower() in self.vocab[label]:
					sum += math.log10(self.vocab[label][word.lower()])
				else:
					sum += math.log10(1 / (self.freq[counter] + self.v + 1))

			sum += math.log10(self.n_doc[label] / self.n_doc_total)
			CNB.append(sum)
			
		return Labels(CNB.index(max(CNB)))


def main(train_split):
	nb = NaiveBayes()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	nb.train(ds)
	
	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(nb, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(nb, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(nb, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
