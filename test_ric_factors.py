from dolfin import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import scipy

import dolfin_to_nparrays as dtn
import cont_obs_utils as cou
import os

u = cou.L2abLinBas(2,8)
Mu = u.massmat()

Chf = scipy.linalg.cholesky(Mu.todense())
Chfi = np.linalg.inv(Chf)

print np.allclose(Mu*np.dot(Chfi, Chfi.T), np.eye(8))

