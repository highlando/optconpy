from scipy.io import savemat, loadmat
import numpy as np

N = 50
arand = np.random.rand(N,N)

timeit np.save('arand',arand)
timeit arand = np.load('arand.npy')

