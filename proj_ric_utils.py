import numpy as np
import scipy.sparse as sps

def solve_proj_lyap_stein(A, B, W, M=None):
    """ approximates X that solves the projected lyap equation

        A*XM + M*XA + B*YM + M*Y*B = -WW*

        BXM = 0    and    M*XB* = 0 

    by considering the equivalent Stein eqns
    and computing the first members of the 
    series converging to X
    """

    ms = [-1]

    def comp_Am11invW(A, M, B, ms, W):
        Np, NW = B.shape[0], W.shape[1]
        SysM = sps.vstack([sps.hstack([A.T + ms.conjugate()*M.T, -B.T]),
                           sps.hstack([B,sps.csr_matrix((Np,Np))])],
                                format='csr')
        Zp = spsla.spsolve(SysM,np.vstack([W, np.zeros((Np,NW))]))
        return Zp[:-Np,:]

    def comp_Im11Z(A, M, B, ms, Z):
        Np, NW = B.shape[0], W.shape[1]
        SysM = sps.vstack([sps.hstack([A.T + ms.conjugate()*M.T, -B.T]),
                           sps.hstack([B,sps.csr_matrix((Np,Np))])],
                                format='csr')
        Zp = spsla.spsolve(SysM,np.vstack([(A.T-ms*M.T)*Z, 
                                            np.zeros((Np,NW))]))
        return Zp[:-Np,:]

    Z = comp_Am11invW(A, M, B, ms, W)
    U = Z

    for n in range(4):
        Z = comp_Im11Z(A, M, B, ms, Z)
        U = np.hstack([U,Z])
        rel_err = np.norm(Z)/np.norm(U)

    U = np.sqrt(-2*ms.real)*U





