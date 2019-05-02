
import scipy
test_count

load_sparse[]

scipy.savez('sparse_matrix.npz', test_count)
sparse_matrix = scipy.load('sparse_matrix.npz')
npzfile = np.load('sparse_matrix.npz')
npzfile=npzfile['arr_0']
train_count=npzfile.item()
















from data_utils import readfile
import numpy as np 

test=readfile("test")

