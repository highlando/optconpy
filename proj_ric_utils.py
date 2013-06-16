import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

def solve_proj_lyap_stein(At=None, B=None, W=None, Mt=None):
    """ approximates X that solves the projected lyap equation

        A*XM + M*XA + B*YM + M*Y*B = -WW*

        BXM = 0    and    M*XB* = 0 

    by considering the equivalent Stein eqns
    and computing the first members of the 
    series converging to X

    At, Mt ... is A*, M* - no transposing in this function
    """

    ms = [-1]

    def app_projinvz(At, Mt, B, ms, Z):
        Np, NZ = B.shape[0], Z.shape[1]
        sysm = sps.vstack([sps.hstack([At + ms.conjugate()*Mt, -B.T]),
                           sps.hstack([B,sps.csr_matrix((Np,Np))])],
                                format='csc')
        
        # raise Warning('TODO: debug') 
        aminv = spsla.splu(sysm)
        # Zp = aminv.solve(np.vstack([Z, np.zeros((Np,NZ))]))
        Zp = Z
        return Zp #[:-Np,:]

    Z = app_projinvz(At, Mt, B, ms[0], W)
    U = Z

    for n in range(4):
        Z = (At - ms[0]*Mt)*Z
        Z = app_projinvz(At, Mt, B, ms[0], Z)
        U = np.hstack([U,Z])
        rel_err = np.linalg.norm(Z)/np.linalg.norm(U)
        print rel_err

    U = np.sqrt(-2*ms[0].real)*U





