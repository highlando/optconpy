import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import lin_alg_utils as lau

n = 500
k = 15
A = 30 * sps.eye(n) + \
    sps.rand(n, n, format='csr')
U = np.random.randn(n, k)
V = np.random.randn(k, n)
Z = np.random.randn(n, k + 2)
Vsp = sps.rand(k, n)

# check the branch with direct solves
AuvInvZ = lau.app_smw_inv(A, umat=U, vmat=V, rhsa=Z, Sinv=None)
AAinvZ = A * AuvInvZ - np.dot(U, np.dot(V, AuvInvZ))


# check the branch where A comes as LU
alusolve = spsla.factorized(A)
AuvInvZ = lau.app_smw_inv(alusolve, umat=U, vmat=V,
                          rhsa=Z, Sinv=None)
AAinvZ = A * AuvInvZ - np.dot(U, np.dot(V, AuvInvZ))
