import numpy as np

def diff_norm(X, Y):
	"""Returns an array with the euclidean norm of the differences between the rows of X and the rows of Y"""
	return np.sqrt(np.sum((X - Y)**2,1))