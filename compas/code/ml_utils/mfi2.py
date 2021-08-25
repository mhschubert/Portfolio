##code author Marina Höhne
##paper: https://arxiv.org/pdf/1611.07567.pdf

from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np

def mfi(x, y, metric='rbf', interaction = [], only_inter = False, degree=2, n_jobs=None):
	'''
	:param x: random variable 1 (e.g. feature matrix)
	:param y: random variable 2 (e.g. prediction scores)
	:param metric: kernel, for instance 'rbf', 'poly'
	:param degree: degree of polynomial kernel (if metric = 'poly')
	:param interaction: list of lists of feature indices to calculate kernel matrix over
	:param n_jobs: scikit-learn multiprocessing/threading parameter (-1 for all processors)
	:return: heatmap of feature importances
	'''
	#check if pandas
	if str(type(y)) == "<class 'pandas.core.frame.DataFrame'>":
		y = y.to_numpy()

	if str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
		x = x.to_numpy()

	if not only_inter:
		N = list(range(len(x[0]))) + interaction #number of features
	else:
		N = interaction
	O = y.shape[1] # number of outputs
	mfi = np.zeros((O, len(N)))
    
    #would be better to parallelize outer and inner loop with joblib instead of njobs
	for j in range(O):
		# KERNEL MATRIX FOR S(X)
		#parallized kernels
		if metric in ["poly", "polynomial"]:
			km_s = pairwise_kernels(np.reshape(y[:, j], (len(y), 1)), Y=None, metric=metric, degree=degree, n_jobs=n_jobs)
		else:
			km_s = pairwise_kernels(np.reshape(y[:, j], (len(y), 1)), Y=None, metric=metric, n_jobs=n_jobs)
		# NORMALIZE
		row_sums = km_s.sum(axis=1)

		km_s = km_s / row_sums[:, np.newaxis]

		# KERNEL MATRIX FOR X_ij OF ALL SAMPLES

		for i in range(0, len(N)):
			if metric in ["poly", "polynomial"]:
				km_x = pairwise_kernels(np.reshape(x[:, N[i]], (x.shape[0], -1)), Y=None,
					metric=metric, degree=degree, n_jobs=n_jobs)
			else:
				km_x = pairwise_kernels(np.reshape(x[:, N[i]], (x.shape[0],-1)), Y=None,
					metric=metric, n_jobs=n_jobs)

			row_sums = km_x.sum(axis=1)

			km_x = (km_x) / row_sums[:, np.newaxis]
			#we only need the trace so we use a running sum instead /einsum for efficient memory management (speedup ~ x100)
			#A = np.dot(km_s, km_x)
			#mfi[i] = np.trace(A)
			mfi[j,i] = np.einsum('ij,ji->', km_s, km_x)

	return mfi

