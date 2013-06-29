import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import linsolv_utils

def solve_proj_lyap_stein(At=None, J=None, W=None, Mt=None, 
                          U=None, V=None, nadisteps=5):
    """ approximates X that solves the projected lyap equation

        A.T*X*M + M.T*X*A + J.T*Y*M + M.T*Y.T*J = -W*W.T

        J*X*M = 0    and    M.T*X*J.T = 0 

    by considering the equivalent Stein eqns
    and computing the first members of the 
    series converging to X

    At, Mt ... is A.T, M.T - no transposing in this function
    """

    ms = [-10]
    NZ = W.shape[0]

    def get_aminv(At, Mt, J, ms):
        """compute the LU of the projection matrix 

        """
        Np = J.shape[0]
        sysm = sps.vstack([sps.hstack([At + ms.conjugate()*Mt, -J.T]),
                           sps.hstack([J,sps.csr_matrix((Np,Np))])],
                                format='csc')
        return spsla.splu(sysm)

    def get_Sinv_smw(Alu,U,V):
        """ compute (the small) inverse of I-V.T*Ainv*U
        """
        aiu = np.zeros(U.shape)
        for ccol in range(U.shape[1]):
            aiu[:,ccol] = Alu.solve(U[:,ccol])
        return np.linalg.inv(np.eye(U.shape[1])-np.dot(V.T,aiu))


    def app_inv_via_smw(Alu, U, V, rhs, Sinv=None):
        """compute the sherman morrison woodbury inverse 

        of A - np.dot(U,V.T) applied to rhs. 
        """
        if Sinv is None:
            Sinv = get_Sinv_smw(Alu,U,V)

        auvirhs = np.zeros(rhs.shape)
        for rhscol in range(rhs.shape[1]):
            crhs = rhs[:,rhscol]
            # the corrected rhs: (I + U*Sinv*VT*Ainv)*rhs
            crhs = crhs + np.dot(U, np.dot(Sinv, 
                                        np.dot(V.T, Alu.solve(crhs))))
            auvirhs[:,rhscol] = Alu.solve(crhs)

        return auvirhs


    def _app_projinvz(Z, At=None, Mt=None, J=None, ms=None, aminv=None):

        if aminv is None:
            aminv = get_aminv(At, Mt, J, ms)

        NZ = Z.shape[0]

        Zp = np.zeros(Z.shape)
        zcol = np.zeros(NZ+J.shape[0])
        for ccol in range(Z.shape[1]):
            zcol[:NZ] = Z[:NZ,ccol]
            Zp[:,ccol] = aminv.solve(zcol)[:NZ]

        return Zp 

    if U is not None and V is not None:
        # to apply the smw formula
        Alu = get_aminv(At, Mt, J, ms[0])

        Ue = np.vstack([U, np.zeros((J.shape[0], U.shape[1]))])
        Ve = np.vstack([V, np.zeros((J.shape[0], V.shape[1]))])
        We = np.vstack([W, np.zeros((J.shape[0], W.shape[1]))])

        Sinv = linsolv_utils.get_Sinv_smw(Alu, U=Ue, V=Ve)
        Z = linsolv_utils.app_smw_inv(Alu, U=Ue, V=Ve, 
                                      rhsa=We, Sinv=Sinv)[:NZ,:]
        for n in range(nadisteps):
            Z = (At - ms[0]*Mt)*Z
            # raise Warning('TODO: debug') 
            Ze = np.vstack([W, np.zeros((J.shape[0], W.shape[1]))])
            Z = linsolv_utils.app_smw_inv(Alu, U=Ue, V=Ve, 
                                          rhsa=Ze, Sinv=Sinv)[:NZ,:]
            print Z.shape
            print U.shape
            U = np.hstack([U,Z])
            raise Warning('TODO: debug') 
            rel_err = np.linalg.norm(Z)/np.linalg.norm(U)
            print rel_err

        return np.sqrt(-2*ms[0].real)*U


    else:
        Z = _app_projinvz(W, At=At, Mt=Mt, J=J, ms=ms[0])
        U = Z

        for n in range(nadisteps):
            Z = (At - ms[0]*Mt)*Z
            # raise Warning('TODO: debug') 
            Z = _app_projinvz(Z, At=At, Mt=Mt, J=J, ms=ms[0])
            print Z.shape
            print U.shape
            U = np.hstack([U,Z])
            rel_err = np.linalg.norm(Z)/np.linalg.norm(U)
            print rel_err

        return np.sqrt(-2*ms[0].real)*U

def get_mTzzTg(MT, Z, tB):
    """ compute the lyapunov coefficient related to the linearization

    TODO: 
    - sparse or dense
    - return just a factor
    """
    return (MT*(np.dot(Z,(Z.T*tB))))*tB.T

def get_mTzzTtb(MT, Z, tB, output=None):
    """ compute the left factor of the lyapunov coefficient 

    related to the linearization
    TODO: 
    - sparse or dense
    """
    if output == 'dense':
        return MT*(np.dot(Z,(Z.T*tB)))
    else:
        return MT*(np.dot(Z,(Z.T*tB)))





