import numpy as np
import threading
import sys
import pkg_resources
pkg_resources.require("Scipy==1.7.0")
import scipy

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def batch_generator(X, y=None, batch_size=1, shuffle=False):
    np.random.seed(123456)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        if type(X) == scipy.sparse.csr.csr_matrix:
            X_batch = X[batch_index,:].todense()
        else:
            X_batch = X[batch_index, :]
        if type(y) != type(None):
            y_batch = y[batch_index,:]
        counter += 1
        if type(y) != type(None):
            yield X_batch, y_batch
        else:

            yield X_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
