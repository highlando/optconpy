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

# the Leray projector
MinvJt = lau.app_luinv_to_spmat(self.Mlu, self.J.T)
Sinv = np.linalg.inv(self.J*MinvJt)
self.P = np.eye(self.NV)-np.dot(MinvJt,Sinv*self.J)

Z = pru.proj_alg_ric_newtonadi(mmat=M, fmat=F, jmat=J, bmat=bmat, 
                            wmat=W, z0=bmat, 
                            nwtn_adi_dict=nwtn_adi_dict)

MtXM = M.T*np.dot(Z,Z.T)*M
MtXb = M.T*np.dot(np.dot(Z, Z.T), bmat)

FtXM = (F.T-uvs.T)*np.dot(Z,Z.T)*M
PtW = np.dot(P.T,W)


ProjRes = np.dot(P.T, np.dot(FtXM, P)) + \
        np.dot(np.dot(P.T, FtXM.T), P) -\
        np.dot(MtXb, MtXb.T) + \
        np.dot(PtW,PtW.T) 

## TEST: result is 'projected' - riccati sol
assertTrue(np.allclose(MtXM,
                        np.dot(P.T,np.dot(MtXM,P))))

## TEST: check projected residual - riccati sol
print 'ric proj res {0}'format(np.linalg.norm(ProjRes))
