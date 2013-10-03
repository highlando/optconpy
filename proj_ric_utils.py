import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import lin_alg_utils as lau

def solve_proj_lyap_stein(A=None, J=None, W=None, M=None, 
                           v=None, u=None, 
                           nadisteps=5,
                           transposed=False):

    """ approximates X that solves the projected lyap equation

        [A-UV].T*X*M + M.T*X*[A-UV] + J.T*Y*M + M.T*Y.T*J = -W*W.T

        J*X*M = 0    and    M.T*X*J.T = 0 

    by considering the equivalent Stein eqns
    and computing the first members of the 
    series converging to X

    We use the SMW formula: 
    (A-UV).-1 = A.-1 + A.-1*U [ I - V A.-1 U].-1 A.-1

    for the transpose:

    (A-UV).-T = A.-T + A.-T*Vt [ I - Ut A.-T Vt ].-1 A.-T 

              = (A.T - Vt Ut).-1

    see numOptAff.pdf
    """

    if transposed:
        At, Mt, ut, vt = A, M, u, v 
    else:
        At, Mt, ut, vt = A.T, M.T, u, v
        if u is not None and v is not None:
            ut, vt = u.T, v.T

    ms = [-10]       
    NZ = W.shape[0]

    def get_atmtlu(At, Mt, J, ms):
        """compute the LU of the projection matrix 

        """
        NP = J.shape[0]
        sysm = sps.vstack([sps.hstack([At + ms.conjugate()*Mt, -J.T]),
                           sps.hstack([J,sps.csr_matrix((NP, NP))])],
                                format='csc')
        return spsla.splu(sysm)

    def _app_projinvz(Z, At=None, Mt=None, 
                        J=None, ms=None, atmtlu=None):

        if atmtlu is None:
            atmtlu = get_atmtlu(At, Mt, J, ms)

        NZ = Z.shape[0]

        Zp = np.zeros(Z.shape)
        zcol = np.zeros(NZ+J.shape[0])
        for ccol in range(Z.shape[1]):
            zcol[:NZ] = Z[:NZ,ccol]
            Zp[:,ccol] = atmtlu.solve(zcol)[:NZ]

        return Zp, atmtlu

    if ut is not None and vt is not None:
        # preps to apply the smw formula
        atmtlu = get_atmtlu(At, Mt, J, ms[0])

        # adding zeros to the coefficients to fit the
        # saddle point systems
        vte = np.vstack([vt, np.zeros((J.shape[0], vt.shape[1]))])
        ute = np.hstack([ut, np.zeros((ut.shape[0], J.shape[0]))])
        We = np.vstack([W, np.zeros((J.shape[0], W.shape[1]))])

        Stinv = lau.get_Sinv_smw(atmtlu, U=vte, V=ute)

        ## Start the ADI iteration

        # debug
        uvst = sps.csr_matrix(np.dot(vt,ut))
        Z = _app_projinvz(W, At=At-uvst, Mt=Mt, 
                          J=J, ms=ms[0])[0]
        # /debug
        print 'SMW norm of iniv: {0}'.format(np.linalg.norm(Z))

        Z = lau.app_smw_inv(atmtlu, U=vte, V=ute, 
                                      rhsa=We, Sinv=Stinv)[:NZ,:]

        print 'SMW norm of iniv: {0}'.format(np.linalg.norm(Z))

        ufac = Z

        for n in range(nadisteps):
            Z = (At - ms[0]*Mt)*Z - np.dot(vt, np.dot(ut, Z))
            # print 'SMW norm {0} of Z: {1}'.format(n, np.linalg.norm(Z))
            Ze = np.vstack([Z, np.zeros((J.shape[0], W.shape[1]))])
            # Z = _app_projinvz(Z, At=At-uvst, Mt=Mt, 
            #                   J=J, ms=ms[0])[0]
            # print 'SMW norm {0} of Zn: {1}'.format(n, np.linalg.norm(Z))
            Z = lau.app_smw_inv(atmtlu, U=vte, V=ute, 
                                          rhsa=Ze, Sinv=Stinv)[:NZ,:]
            # print 'SMW norm {0} of Zn: {1}'.format(n, np.linalg.norm(Z))
            ufac = np.hstack([ufac, Z])

        rel_err = np.linalg.norm(Z)/np.linalg.norm(ufac)
        print 'Number of ADI steps {0} - Relative norm of the update {1}'.format(nadisteps, rel_err)

        return np.sqrt(-2*ms[0].real)*ufac

    else:
        Z, atmtlu = _app_projinvz(W, At=At, Mt=Mt, J=J, ms=ms[0])
        # print 'CLA norm of iniv: {0}'.format(np.linalg.norm(Z))
        ufac = Z

        for n in range(nadisteps):
            Z = (At - ms[0]*Mt)*Z
            # print 'CLA norm {0} of Z: {1}'.format(n, np.linalg.norm(Z))
            Z = _app_projinvz(Z, At=At, Mt=Mt, 
                              J=J, ms=ms[0])[0]
            # print 'CLA norm {0} of Zn: {1}'.format(n, np.linalg.norm(Z))
            ufac = np.hstack([ufac,Z])

        rel_err = np.linalg.norm(Z)/np.linalg.norm(ufac)
        print 'Number of ADI steps {0} - Relative error in the update {1}'.format(nadisteps, rel_err)

        return np.sqrt(-2*ms[0].real)*ufac

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


def proj_alg_ric_newtonadi(mmat=None, fmat=None, jmat=None, bmat=None, 
                            wmat=None, z0=None, 
                            newtonadisteps=10, adisteps=100,
                            transposed=False):

    """ solve the projected algebraic ricc via newton adi 

    M.T*X*F + F.T*X*M - M.T*X*B*B.T*X*M + J(Y) = -WW.T

        JXM = 0 and M.TXJ.T = 0

    """

    if transposed:
        mt, ft  = mmat, fmat
    else:
        mt, ft  = mmat.T, fmat.T
        transposed = True
        
    znn = solve_proj_lyap_stein(A=ft,
                                M=mt,
                                J=jmat,
                                W=wmat,
                                nadisteps=adisteps,
                                transposed=transposed)

    znc = znn

    for nnwtadi in range(newtonadisteps):

        mtxb = mt*np.dot(znc, np.dot(znc.T, bmat))
        rhsadi = np.hstack([mtxb, wmat])
        rhsadi = wmat

        # to avoid a dense matrix we use the smw formula
        # to compute (A-UV).-1
        # for the factorization mTxg = mTxtb * tbT = U*V

        znn = solve_proj_lyap_stein(A=ft,
                                    M=mt,
                                    # U=mtxb, V=bmat.T,
                                    J=jmat,
                                    W=rhsadi,
                                    nadisteps=adisteps,
                                    transpose=transpose)
        
        fndif = lau.comp_frobnorm_factored_difference(znc, znn)
        print np.sqrt(fndif)
        print 'current f norm of newton adi update is {0}'.format(fndif)
        znc = znn



