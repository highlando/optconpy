import sympy as smp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import proj_ric_utils as pru

Nv = 250
Np = 40
Ny = 5 
nwtn_adi_dict = dict(
            adi_max_steps=250,
            adi_newZ_reltol=4e-8,
            nwtn_max_steps=24,
            nwtn_upd_reltol=4e-8,
            nwtn_upd_abstol=4e-8
                    )

# -F, M spd -- coefficient matrices
F = -sps.eye(Nv) - sps.rand(Nv, Nv)*sps.rand(Nv, Nv) 
M = sps.eye(Nv) + sps.rand(Nv, Nv)*sps.rand(Nv, Nv) 

# right-handside: C= -W*W.T
W = np.random.randn(Nv, Ny)
U = np.random.randn(Nv, Ny)
V = np.random.randn(Nv, Ny) 
bmat = np.random.randn(Nv, Ny+2)

# we need J sparse and of full rank
for auxk in range(5):
    try:
        J = sps.rand(Np, Nv, density=0.03, format='csr')
        spsla.splu(J*J.T)
        break
    except RuntimeError:
        print 'J not full row-rank.. I make another try'
try:
    spsla.splu(J*J.T)
except RuntimeError:
    raise Warning('Fail: J is not full rank')

U = 1e-4*U
V = V.T
print np.linalg.norm(U)

# Z = pru.solve_proj_lyap_stein(A=F, M=M, 
#                                 umat=U, vmat=V, 
#                                 J=J, W=W,
#                                 nadisteps=adisteps)
# 
# uvs = sps.csr_matrix(np.dot(U,V))
# Z2 = pru.solve_proj_lyap_stein(A=F-uvs, M=M, 
#                                 J=J, W=W,
#                                 nadisteps=adisteps)
# 
# print 'this should be 0={0}'.format(np.linalg.norm(Z-Z2))

Z = pru.proj_alg_ric_newtonadi(mmat=M, fmat=F, jmat=J, bmat=bmat, 
                            wmat=W, z0=bmat, 
                            nwtn_adi_dict=nwtn_adi_dict)

