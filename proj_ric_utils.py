import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import lin_alg_utils as lau

def solve_proj_lyap_stein(A=None, J=None, W=None, M=None, 
                           umat=None, vmat=None, 
                           transposed=False,
                           adi_dict=dict(
                                adi_max_steps=150,
                                adi_newZ_reltol=1e-8
                                        )
                           ):

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
        At, Mt = A, M 
    else:
        At, Mt = A.T, M.T

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


    adi_step = 0
    rel_Z_err = 2

    if umat is not None and vmat is not None:
        # preps to apply the smw formula
        atmtlu = get_atmtlu(At, Mt, J, ms[0])

        # adding zeros to the coefficients to fit the
        # saddle point systems
        vmate = np.hstack([vmat, np.zeros((vmat.shape[0], J.shape[0]))])
        umate = np.vstack([umat, np.zeros((J.shape[0], umat.shape[1]))])
        We = np.vstack([W, np.zeros((J.shape[0], W.shape[1]))])


        Stinv = lau.get_Sinv_smw(atmtlu, umat=vmate.T, vmat=umate.T)

        ## Start the ADI iteration

        Z = lau.app_smw_inv(atmtlu, umat=vmate.T, vmat=umate.T, 
                                      rhsa=We, Sinv=Stinv)[:NZ,:]

        ufac = Z
        while adi_step < adi_dict['adi_max_steps'] and \
              rel_Z_err > adi_dict['adi_newZ_reltol']:

            Z = (At - ms[0]*Mt)*Z - np.dot(vmat.T, np.dot(umat.T, Z))
            Ze = np.vstack([Z, np.zeros((J.shape[0], W.shape[1]))])
            Z = lau.app_smw_inv(atmtlu, umat=vmate.T, vmat=umate.T, 
                                          rhsa=Ze, Sinv=Stinv)[:NZ,:]
            ufac = np.hstack([ufac, Z])
            rel_Z_err = np.linalg.norm(Z)/np.linalg.norm(ufac)

            adi_step += 1

        print ('Number of ADI steps {0} -- \n' + 
                'Relative norm of the update {1}'
                    ).format(adi_step, rel_Z_err)

        return np.sqrt(-2*ms[0].real)*ufac

    else:
        Z, atmtlu = _app_projinvz(W, At=At, Mt=Mt, J=J, ms=ms[0])
        ufac = Z

        while adi_step < adi_dict['adi_max_steps'] and \
              rel_Z_err > adi_dict['adi_newZ_reltol']:

            Z = (At - ms[0]*Mt)*Z
            Z = _app_projinvz(Z, At=At, Mt=Mt, 
                              J=J, ms=ms[0])[0]
            ufac = np.hstack([ufac,Z])
            rel_Z_err = np.linalg.norm(Z)/np.linalg.norm(ufac)

            adi_step += 1

        print ('Number of ADI steps {0} -- \n' + 
                'Relative norm of the update {1}'
                    ).format(adi_step, rel_Z_err)

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

def proj_alg_ric_newtonadi(mmat=None, fmat=None, jmat=None, 
                            bmat=None, wmat=None, z0=None, 
                            transposed=False,
                            nwtn_adi_dict=dict(
                                        adi_max_steps=150,
                                        adi_newZ_reltol=1e-8,
                                        nwtn_max_steps=14,
                                        nwtn_upd_reltol=1e-12
                                                )
                            ):

    """ solve the projected algebraic ricc via newton adi 

    M.T*X*F + F.T*X*M - M.T*X*B*B.T*X*M + J(Y) = -WW.T

        JXM = 0 and M.TXJ.T = 0

    """

    if transposed:
        mt, ft  = mmat, fmat
    else:
        mt, ft  = mmat.T, fmat.T
        transposed = True
        
    znc = z0
    nwtn_stp = 0

    while nwtn_stp < adi_dict['nwtn_max_steps'] and \
          rel_upd_fnorm > adi_dict['nwtn_upd_reltol']:

        mtxb = mt*np.dot(znc, np.dot(znc.T, bmat))
        rhsadi = np.hstack([mtxb, wmat])

        # to avoid a dense matrix we use the smw formula
        # to compute (A-UV).-T
        # for the factorization mTxg.T =  tb * mTxtb = U*V

        znn = solve_proj_lyap_stein(A=ft, M=mt, J=jmat,
                                    W=rhsadi,
                                    umat=bmat, vmat=mtxb.T,
                                    transposed=transposed,
                                    adi_dict=nwtn_adi_dict)

        fndif, fnznc, fnznn = \
                lau.comp_sqrdfrobnorm_factored_difference(znn, znc,
                                                    ret_sing_norms=True)

        rel_upd_fnorm = np.sqrt(fndif/fnznn)
        print '\ncurrent f norm of newton adi update is {0}\n'.format(rel_upd_fnorm)

        znc = znn
        nwtn_stp += 1


