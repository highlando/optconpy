import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import lin_alg_utils as lau


def solve_proj_lyap_stein(A=None, J=None, W=None, M=None,
                          umat=None, vmat=None,
                          transposed=False,
                          adi_dict=dict(adi_max_steps=150,
                                        adi_newZ_reltol=1e-8)
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

    ms = [-10, -10, -9]
    NZ = W.shape[0]

    def get_atmtlu(At, Mt, J, ms):
        """compute the LU of the projection matrix

        """
        NP = J.shape[0]
        sysm = sps.vstack([sps.hstack([At + ms.conjugate() * Mt, -J.T]),
                           sps.hstack([J, sps.csr_matrix((NP, NP))])],
                          format='csc')
        return spsla.factorized(sysm)

    def _app_projinvz(Z, At=None, Mt=None,
                      J=None, ms=None, atmtlu=None):

        if atmtlu is None:
            atmtlu = get_atmtlu(At, Mt, J, ms)

        NZ = Z.shape[0]

        Zp = np.zeros(Z.shape)
        zcol = np.zeros(NZ + J.shape[0])
        for ccol in range(Z.shape[1]):
            zcol[:NZ] = Z[:NZ, ccol]
            Zp[:, ccol] = atmtlu(zcol)[:NZ]

        return Zp, atmtlu

    adi_step = 0
    rel_newZ_norm = 2
    adi_rel_newZ_norms = []

    if umat is not None and vmat is not None:
        # preps to apply the smw formula
        # adding zeros to the coefficients to fit the
        # saddle point systems
        vmate = np.hstack([vmat, np.zeros((vmat.shape[0], J.shape[0]))])
        if sps.isspmatrix(umat):
            umate = sps.vstack([umat, sps.csr_matrix((J.shape[0],
                                                     umat.shape[1]))])
        else:
            umate = np.vstack([umat, np.zeros((J.shape[0], umat.shape[1]))])

        We = np.vstack([W, np.zeros((J.shape[0], W.shape[1]))])
        atmtlulist = []
        stinvlist = []
        for mu in ms:
            atmtlumu = get_atmtlu(At, Mt, J, mu)
            atmtlulist.append(atmtlumu)
            stinvlist.append(lau.get_Sinv_smw(atmtlumu,
                                              umat=vmate.T, vmat=umate.T))

        #  Start the ADI iteration

        Z = lau.app_smw_inv(atmtlulist[0], umat=vmate.T, vmat=umate.T,
                            rhsa=np.sqrt(-2 * ms[0].real) * We,
                            Sinv=stinvlist[0])[:NZ, :]

        ufac = Z
        u_norm_sqrd = np.linalg.norm(Z) ** 2

        muind = 1

        while adi_step < adi_dict['adi_max_steps'] and \
                rel_newZ_norm > adi_dict['adi_newZ_reltol']:

            Ze = np.vstack([Mt*Z, np.zeros((J.shape[0], W.shape[1]))])
            Zi = lau.app_smw_inv(atmtlulist[muind], umat=vmate.T,
                                 vmat=umate.T, rhsa=Ze,
                                 Sinv=stinvlist[muind])[:NZ, :]

            Z = np.sqrt(ms[muind].real / ms[muind-1].real) *\
                (Z - (ms[muind] + ms[muind-1].conjugate()) * Zi)

            z_norm_sqrd = np.linalg.norm(Z) ** 2
            u_norm_sqrd = u_norm_sqrd + z_norm_sqrd

            ufac = np.hstack([ufac, Z])
            rel_newZ_norm = np.sqrt(z_norm_sqrd / u_norm_sqrd)
            # np.linalg.norm(Z)/np.linalg.norm(ufac)

            adi_step += 1
            muind = np.mod(muind+1, len(ms))
            adi_rel_newZ_norms.append(rel_newZ_norm)

        try:
            if adi_dict['verbose']:
                print ('Number of ADI steps {0} -- \n' +
                       'Relative norm of the update {1}'
                       ).format(adi_step, rel_newZ_norm)
                print 'sqrd norm of Z: {0}'.format(u_norm_sqrd)
        except KeyError:
            pass  # no verbosity specified - nothing is shown

    else:
        for mu in ms:
            atmtlumu = get_atmtlu(At, Mt, J, mu)

        Z = _app_projinvz(W, At=At, Mt=Mt, J=J, ms=ms[0])
        ufac = np.sqrt(-2 * ms[0].real) * Z
        u_norm_sqrd = np.linalg.norm(Z) ** 2

        while adi_step < adi_dict['adi_max_steps'] and \
                rel_newZ_norm > adi_dict['adi_newZ_reltol']:

            Z = (At - ms[0] * Mt) * Z
            Z = _app_projinvz(Z, atmtlu=atmtlu, Mt=Mt,
                              J=J, ms=ms[0])[0]
            ufac = np.hstack([ufac, np.sqrt(-2 * ms[0].real) * Z])

            z_norm_sqrd = np.linalg.norm(Z) ** 2
            u_norm_sqrd = u_norm_sqrd + z_norm_sqrd

            rel_newZ_norm = np.sqrt(z_norm_sqrd / u_norm_sqrd)

            adi_step += 1
            adi_rel_newZ_norms.append(rel_newZ_norm)

        try:
            if adi_dict['verbose']:
                print ('Number of ADI steps {0} -- \n' +
                       'Relative norm of the update {1}'
                       ).format(adi_step, rel_newZ_norm)
        except KeyError:
            pass  # no verbosity specified - nothing is shown

    return dict(zfac=ufac,
                adi_rel_newZ_norms=adi_rel_newZ_norms)


def get_mTzzTg(MT, Z, tB):
    """ compute the lyapunov coefficient related to the linearization

    TODO:
    - sparse or dense
    - return just a factor
    """
    return (MT * (np.dot(Z, (Z.T * tB)))) * tB.T


def get_mTzzTtb(MT, Z, tB, output=None):
    """ compute the left factor of the lyapunov coefficient

    related to the linearization
    TODO:
    - sparse or dense
    """
    if output == 'dense':
        return MT * (np.dot(Z, (Z.T * tB)))
    else:
        return MT * (np.dot(Z, (Z.T * tB)))


def proj_alg_ric_newtonadi(mmat=None, fmat=None, jmat=None,
                           bmat=None, wmat=None, z0=None,
                           transposed=False,
                           nwtn_adi_dict=dict(adi_max_steps=150,
                                              adi_newZ_reltol=1e-8,
                                              nwtn_max_steps=14,
                                              nwtn_upd_reltol=1e-8)):
    """ solve the projected algebraic ricc via newton adi

    M.T*X*F + F.T*X*M - M.T*X*B*B.T*X*M + J(Y) = -WW.T

        JXM = 0 and M.TXJ.T = 0

    """

    if transposed:
        mt, ft = mmat, fmat
    else:
        mt, ft = mmat.T, fmat.T
        transposed = True

    znc = z0
    nwtn_stp, upd_fnorm = 0, 2
    nwtn_upd_fnorms = []

    while nwtn_stp < nwtn_adi_dict['nwtn_max_steps'] and \
            upd_fnorm > nwtn_adi_dict['nwtn_upd_abstol']:

        try:
            mtxb = mt * np.dot(znc, np.dot(znc.T, bmat))
        except ValueError:  # if bmat is sparse
            mtxb = mt * np.dot(znc, znc.T * bmat)

        rhsadi = np.hstack([mtxb, wmat])

        # to avoid a dense matrix we use the smw formula
        # to compute (A-UV).-T
        # for the factorization mTxg.T =  tb * mTxtb = U*V

        znn = solve_proj_lyap_stein(A=ft, M=mt, J=jmat,
                                    W=rhsadi,
                                    umat=bmat, vmat=mtxb.T,
                                    transposed=transposed,
                                    adi_dict=nwtn_adi_dict)['zfac']

        if nwtn_adi_dict['full_upd_norm_check']:
            upd_fnorm = lau.comp_sqfnrm_factrd_diff(znn, znc)
            upd_fnorm = np.sqrt(np.abs(upd_fnorm))

        else:
            vec = np.random.randn(znn.shape[0], 1)
            vecn1 = comp_diff_zzv(znn, znc, vec)
            vec = np.random.randn(znn.shape[0], 1)
            vecn2 = comp_diff_zzv(znn, znc, vec)
            if vecn2 + vecn1 < nwtn_adi_dict['nwtn_upd_abstol']:
                znred = compress_Z(znn, 500)
                zcred = compress_Z(znc, 500)
                upred_fnorm = lau.comp_sqfnrm_factrd_diff(znred, zcred)
                print 'shapes', znn.shape, znred.shape
                print 'comp upd norms', upd_fnorm, upred_fnorm
                print vecn2+vecn1, znc.shape
                upd_fnorm = lau.comp_sqfnrm_factrd_diff(znn, znc)
                upd_fnorm = np.sqrt(np.abs(upd_fnorm))

        nwtn_upd_fnorms.append(upd_fnorm)

        try:
            if nwtn_adi_dict['verbose']:
                print ('Newton ADI step: {1} --' +
                       'f norm of update: {0}').format(upd_fnorm,
                                                       nwtn_stp + 1)
                if not nwtn_adi_dict['full_upd_norm_check']:
                    print ('btw... we used an estimated norm:').\
                        format(nwtn_stp + 1)
                    print '|| upd * vec || / || vec || = {0}'.format(vecn1)
                    print '|| upd * vec || / || vec || = {0}'.format(vecn2)

        except KeyError:
            pass    # no verbosity specified - nothing is shown

        znc = znn
        nwtn_stp += 1

    return dict(zfac=znn,
                nwtn_upd_fnorms=nwtn_upd_fnorms)


def compress_Z(Z, k=None, tol=None):
    """routine that compresses the columns Z by means of a truncated SVD

    such that it ZZ.T is still well approximated"""

    nny = Z.shape[1]
    U, s, V = np.linalg.svd(Z, full_matrices=False)

    S = sps.dia_matrix((s, 0), (nny, nny)).tocsr()

    svred = S[:k, :][:, :k] * V[:k, :k]

    return np.dot(U[:, :k], svred)


def comp_diff_zzv(zone, ztwo, vec):
    return np.linalg.norm(np.dot(zone, np.dot(zone.T, vec)) -
                          np.dot(ztwo, np.dot(ztwo.T, vec)))
