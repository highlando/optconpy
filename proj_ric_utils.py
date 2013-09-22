import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import linsolv_utils

def solve_proj_lyap_stein(At=None, J=None, W=None, Mt=None, 
                          U=None, V=None, nadisteps=5):

    """ approximates X that solves the projected lyap equation

        [A-UV].T*X*M + M.T*X*[A-UV] + J.T*Y*M + M.T*Y.T*J = -W*W.T

        J*X*M = 0    and    M.T*X*J.T = 0 

    by considering the equivalent Stein eqns
    and computing the first members of the 
    series converging to X

    At, Mt ... is A.T, M.T - no transposing in this function

    We use the SMW formula: 
    (A-UV).-1 = A.-1 + A.-1*U[I-VA.-1U).-1 A.-1
    see numOptAff.pdf
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
            Ze = np.vstack([W, np.zeros((J.shape[0], W.shape[1]))])
            Z = linsolv_utils.app_smw_inv(Alu, U=Ue, V=Ve, 
                                          rhsa=Ze, Sinv=Sinv)[:NZ,:]
            print Z.shape
            print U.shape
            U = np.hstack([U,Z])
            rel_err = np.linalg.norm(Z)/np.linalg.norm(U)
            print rel_err

        return np.sqrt(-2*ms[0].real)*U


    else:
        Z = _app_projinvz(W, At=At, Mt=Mt, J=J, ms=ms[0])
        U = Z

        for n in range(nadisteps):
            Z = (At - ms[0]*Mt)*Z
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

def comp_frobnorm_factored_difference(zone, ztwo):
    """compute the squared Frobenius norm of z1*z1.T - z2*z2.T

    """
    tr1sq = (zone*zone).sum(-1)
    tr2sq = (ztwo*ztwo).sum(-1)
    tr12  = np.sqrt(tr1sq*tr2sq)

    return (tr1sq - 2*tr12 + tr2sq).sum()

def proj_alg_ric_newtonadi(mt=None, ft=None, jmat=None, bmat=None, 
                            wmat=None, z0=None, 
                            newtonadisteps=10, adisteps=100):

    """ solve the projected algebraic ricc via newton adi 

    M.T*X*F + F.T*X*M - M.T*X*B*B.T*X*M + J(Y) = -WW.T

        JXM = 0 and M.TXJ.T = 0

    """
    znc = z0

    for nnwtadi in range(newtonadisteps):

        mtxb = mt*np.dot(znc, znc.T*B)
        rhsadi = np.hstack([mtxb, wmat])

        # to avoid a dense matrix we use the smw formula
        # to compute (A-UV).-1
        # for the factorization mTxg = mTxtb * tbT = U*V

        znc = solve_proj_lyap_stein(At=ft,
                                    Mt=mt,
                                    U=mTxtb,
                                    V=bmat.todense(),
                                    J=jmat,
                                    W=wmat,
                                    nadisteps=nadisteps)



