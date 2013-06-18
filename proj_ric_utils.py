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

    ms = [-10]

    def get_aminv(At, Mt, B, ms):
        """compute the LU of the projection matrix 

        """
        Np = B.shape[0]
        sysm = sps.vstack([sps.hstack([At + ms.conjugate()*Mt, -B.T]),
                           sps.hstack([B,sps.csr_matrix((Np,Np))])],
                                format='csc')
        return spsla.splu(sysm)

    def _app_projinvz(Z, At=None, Mt=None, B=None, ms=None, aminv=None):

        # raise Warning('TODO: debug') 

        if aminv is None:
            aminv = get_aminv(At, Mt, B, ms)

        NZ = Z.shape[0]

        Zp = np.zeros(Z.shape)
        zcol = np.zeros(NZ+B.shape[0])
        for ccol in range(Z.shape[1]):
            zcol[:NZ] = Z[:NZ,ccol]
            Zp[:,ccol] = aminv.solve(zcol)[:NZ]

        return Zp 

    Z = _app_projinvz(W, At=At, Mt=Mt, B=B, ms=ms[0])
    U = Z

    for n in range(49):
        Z = (At - ms[0]*Mt)*Z
        Z = _app_projinvz(Z, At=At, Mt=Mt, B=B, ms=ms[0])
        print Z.shape
        print U.shape
        U = np.hstack([U,Z])
        rel_err = np.linalg.norm(Z)/np.linalg.norm(U)
        print rel_err

    U = np.sqrt(-2*ms[0].real)*U





