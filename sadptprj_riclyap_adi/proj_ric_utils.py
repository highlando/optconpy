import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import lin_alg_utils as lau


def solve_stst_feedbacknthrough(amat=None, mmat=None, jmat=None,
                                bmat=None, cmat=None,
                                fv=None, fl=None, fg=None, fl2=None,
                                nwtn_adi_dict=dict(adi_max_steps=150,
                                                   adi_newZ_reltol=1e-5,
                                                   nwtn_max_steps=14,
                                                   nwtn_upd_reltol=1e-8)):
    """solve for the stabilizing feedback gain and the feedthrough

    for the linear time invariant case"""

    Z = proj_alg_ric_newtonadi(mmat=mmat, amat=amat, jmat=jmat,
                               bmat=bmat, wmat=cmat,
                               nwtn_adi_dict=nwtn_adi_dict)

    mtxb = get_mTzzTtb(mmat.T, Z, bmat)
    mtxfv = get_mTzzTtb(mmat.T, Z, fv)

    wft = lau.solve_sadpnt_smw(amat=-amat.T, jmat=jmat, rhsv=fl-mtxfv,
                               umat=-mtxb, vmat=bmat.T)

    return Z, wft


def solve_proj_lyap_stein(amat=None, jmat=None, wmat=None, mmat=None,
                          umat=None, vmat=None,
                          transposed=False,
                          adi_dict=dict(adi_max_steps=150,
                                        adi_newZ_reltol=1e-8),
                          nwtn_adi_dict=None, **kw):
    """ approximates the solution X to the projected lyap equation

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
    if nwtn_adi_dict is not None:
        adi_dict = nwtn_adi_dict
        # so we can pass the same dicts to lyap and ric solve

    if transposed:
        At, Mt = amat, mmat
    else:
        At, Mt = amat.T, mmat.T

    # TODO: compute optimal shifts
    try:
        ms = adi_dict['ms']
    except KeyError:
        ms = [-30.0, -20.0, -10.0, -5.0, -3.0, -1.0]

    if adi_dict['verbose']:
        print ('\nAdishifts: {0} ').format(ms)

    NZ = wmat.shape[0]

    def get_atmtlu(At, Mt, jmat, ms):
        """compute the LU of the projection matrix

        """
        NP = jmat.shape[0]
        sysm = sps.vstack([sps.hstack([At + ms.conjugate() * Mt, -jmat.T]),
                           sps.hstack([jmat, sps.csr_matrix((NP, NP))])],
                          format='csc')
        return spsla.factorized(sysm)

    def _app_projinvz(Z, At=None, Mt=None,
                      jmat=None, ms=None, atmtlu=None):

        if atmtlu is None:
            atmtlu = get_atmtlu(At, Mt, jmat, ms)

        NZ = Z.shape[0]

        Zp = np.zeros(Z.shape)
        zcol = np.zeros(NZ + jmat.shape[0])
        for ccol in range(Z.shape[1]):
            if sps.isspmatrix(Z):
                zcol[:NZ] = Z[:NZ, ccol].todense().flatten()
            else:
                zcol[:NZ] = Z[:NZ, ccol]
            Zp[:, ccol] = atmtlu(zcol)[:NZ]

        return Zp, atmtlu

    adi_step = 0
    rel_newZ_norm = 2
    adi_rel_newZ_norms = []

    atmtlulist = []
    for mu in ms:
        atmtlumu = get_atmtlu(At, Mt, jmat, mu)
        atmtlulist.append(atmtlumu)

    if umat is not None and vmat is not None:
        # preps to apply the smw formula
        # adding zeros to the coefficients to fit the
        # saddle point systems
        vmate = np.hstack([vmat, np.zeros((vmat.shape[0], jmat.shape[0]))])
        if sps.isspmatrix(umat):
            umate = sps.vstack([umat, sps.csr_matrix((jmat.shape[0],
                                                     umat.shape[1]))])
        else:
            umate = np.vstack([umat, np.zeros((jmat.shape[0], umat.shape[1]))])

        stinvlist = []
        for ncurmu, mu in enumerate(ms):
            stinvlist.append(lau.get_Sinv_smw(atmtlulist[ncurmu],
                                              umat=vmate.T, vmat=umate.T))

        #  Start the ADI iteration

        We = np.vstack([wmat, np.zeros((jmat.shape[0], wmat.shape[1]))])

        Z = lau.app_smw_inv(atmtlulist[0], umat=vmate.T, vmat=umate.T,
                            rhsa=np.sqrt(-2 * ms[0].real) * We,
                            Sinv=stinvlist[0])[:NZ, :]

        ufac = Z
        u_norm_sqrd = np.linalg.norm(Z) ** 2

        muind = 1
        muind = np.mod(muind, len(ms))

        while adi_step < adi_dict['adi_max_steps'] and \
                rel_newZ_norm > adi_dict['adi_newZ_reltol']:

            Ze = np.vstack([Mt*Z, np.zeros((jmat.shape[0], wmat.shape[1]))])
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
                if adi_dict['check_lyap_res'] and np.mod(adi_step, 10) == 0:
                    sqrdprolyares = \
                        comp_proj_lyap_res_norm(Z, amat=amat, mmat=mmat,
                                                wmat=wmat, jmat=jmat,
                                                umat=umat, vmat=vmat)
                    print 'adistep ', adi_step
                    print 'cur proj lyap res: ', np.sqrt(sqrdprolyares)
                    print 'rel Z norm: \n', rel_newZ_norm
            except KeyError:
                pass  # no such option specified

        try:
            if adi_dict['verbose']:
                print ('Number of ADI steps {0} -- \n' +
                       'Relative norm of the update {1}'
                       ).format(adi_step, rel_newZ_norm)
                print 'sqrd norm of Z: {0}'.format(u_norm_sqrd)
        except KeyError:
            pass  # no verbosity specified - nothing is shown

    else:

        Z = _app_projinvz(np.sqrt(-2*ms[0].real)*wmat, jmat=jmat,
                          atmtlu=atmtlulist[0])[0]

        ufac = Z
        u_norm_sqrd = np.linalg.norm(Z) ** 2
        muind = 1
        muind = np.mod(muind, len(ms))

        while adi_step < adi_dict['adi_max_steps'] and \
                rel_newZ_norm > adi_dict['adi_newZ_reltol']:

            Zi = _app_projinvz(Mt*Z, jmat=jmat, atmtlu=atmtlulist[muind])[0]

            Z = np.sqrt(ms[muind].real / ms[muind-1].real) *\
                (Z - (ms[muind] + ms[muind-1].conjugate()) * Zi)

            ufac = np.hstack([ufac, Z])

            z_norm_sqrd = np.linalg.norm(Z) ** 2
            u_norm_sqrd = u_norm_sqrd + z_norm_sqrd

            rel_newZ_norm = np.sqrt(z_norm_sqrd / u_norm_sqrd)

            adi_step += 1
            muind = np.mod(muind+1, len(ms))
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


def get_mTzzTtb(MT, Z, tB, output=None):
    """ compute the left factor of the lyapunov coefficient

    related to the linearization
    """
    if sps.isspmatrix(tB):
        return MT * (np.dot(Z, Z.T*tB))
    else:
        return MT*(np.dot(Z, np.dot(Z.T, tB)))


def pymess_dae2_newtonadi(mmat=None, amat=None, jmat=None,
                          bmat=None, wmat=None, z0=None, mtxoldb=None,
                          transposed=False,
                          nwtn_adi_dict=dict(adi_max_steps=150,
                                             adi_newZ_reltol=1e-5,
                                             nwtn_max_steps=14,
                                             nwtn_upd_reltol=1e-8),
                          **kw):
    """ solve the projected algebraic ricc via newton adi

    `M.T*X*A + A.T*X*M - M.T*X*B*B.T*X*M + J(Y) = -WW.T`

    `JXM = 0 and M.TXJ.T = 0`

    If `mtxb` is given,
    (e.g. as the feedback computed in a previous step of a Newton iteration),
    the coefficient matrix with feedback

    `A.T <- A.T - mtxb*b`

    is considered

    """

    if transposed:
        mt, at = mmat, amat
    else:
        mt, at = mmat.T, amat.T
    loctransposed = True

    if sps.isspmatrix(wmat):
        wmat = np.array(wmat.todense())

    znc = z0
    nwtn_stp, upd_fnorm, upd_fnorm_n = 0, None, None
    nwtn_upd_fnorms = []

    while nwtn_stp < nwtn_adi_dict['nwtn_max_steps']:

        if znc is None:  # i.e., if z0 was None
            rhsadi = wmat
            mtxbt = None
        else:
            try:
                mtxb = mt * np.dot(znc, np.dot(znc.T, bmat))
            except ValueError:  # if bmat is sparse
                mtxb = mt * np.dot(znc, znc.T * bmat)
            mtxbt = mtxb.T
            rhsadi = np.hstack([mtxb, wmat])
        # to avoid a dense matrix we use the smw formula
        # to compute (A-UV).-T
        # for the factorization mTxg.T =  tb * mTxtb.T = U*V
        # and we add the previous feedback:
        if mtxoldb is not None:
            mtxbt = mtxbt + mtxoldb.T

        znn = solve_proj_lyap_stein(amat=at, mmat=mt, jmat=jmat,
                                    wmat=rhsadi,
                                    umat=bmat, vmat=mtxbt,
                                    transposed=loctransposed,
                                    nwtn_adi_dict=nwtn_adi_dict)['zfac']

        if nwtn_adi_dict['full_upd_norm_check']:
            if znc is None:  # there was no initial guess
                znc = 0*znn
            upd_fnorm = lau.comp_sqfnrm_factrd_diff(znn, znc)
            upd_fnorm = np.sqrt(np.abs(upd_fnorm))

        else:
            if znc is None:  # there was no initial guess
                znc = 0*znn
            vec = np.random.randn(znn.shape[0], 1)
            vecn1 = comp_diff_zzv(znn, znc, vec)
            vec = np.random.randn(znn.shape[0], 1)
            vecn2 = comp_diff_zzv(znn, znc, vec)
            vec = np.random.randn(znn.shape[0], 1)
            # to make the estimate relative
            vecn3 = np.linalg.norm(np.dot(znn, np.dot(znn.T, vec)))
            if (vecn2 + vecn1)/vecn3 < 8e-9:
                upd_fnorm, nzn, nzc = lau.\
                    comp_sqfnrm_factrd_diff(znn, znc, ret_sing_norms=True)
                upd_fnorm_n = np.sqrt(np.abs(upd_fnorm) / np.abs(nzn))

        nwtn_upd_fnorms.append(upd_fnorm_n)
        try:
            if np.allclose(upd_fnorm_n, upd_fnorm):
                print 'no more change in the norm of the update... break'
                break
        except TypeError:
            pass
        if nwtn_adi_dict['full_upd_norm_check']:
            upd_fnorm = upd_fnorm_n
        elif (vecn2 + vecn1)/vecn3 < 8e-9:
            upd_fnorm = upd_fnorm_n

        try:
            if nwtn_adi_dict['verbose']:
                print ('Newton ADI step: {1} -- ' +
                       'rel f norm of update: {0}').format(upd_fnorm,
                                                           nwtn_stp + 1)
                if not nwtn_adi_dict['full_upd_norm_check']:
                    print ('btw, we decided whether to compute the actual ' +
                           'norm on the base of estimates:')
                    print '|| upd * vec || / || vec || = {0}'.format(vecn2)
                    print '||Z*vec|| = {0}'.format(vecn3)

        except KeyError:
            pass    # no verbosity specified - nothing is shown

        znc = znn
        nwtn_stp += 1
        if (upd_fnorm is not None
                and upd_fnorm < nwtn_adi_dict['nwtn_upd_reltol']):
            break

    return dict(zfac=znn, nwtn_upd_fnorms=nwtn_upd_fnorms)


def proj_alg_ric_newtonadi(mmat=None, amat=None, jmat=None,
                           bmat=None, wmat=None, z0=None, mtxoldb=None,
                           transposed=False,
                           nwtn_adi_dict=dict(adi_max_steps=150,
                                              adi_newZ_reltol=1e-5,
                                              nwtn_max_steps=14,
                                              nwtn_upd_reltol=1e-8),
                           **kw):
    """ solve the projected algebraic ricc via newton adi

    `M.T*X*A + A.T*X*M - M.T*X*B*B.T*X*M + J(Y) = -WW.T`

    `JXM = 0 and M.TXJ.T = 0`

    If `mtxb` is given,
    (e.g. as the feedback computed in a previous step of a Newton iteration),
    the coefficient matrix with feedback

    `A.T <- A.T - mtxb*b`

    is considered

    """

    if transposed:
        mt, at = mmat, amat
    else:
        mt, at = mmat.T, amat.T
    loctransposed = True

    if sps.isspmatrix(wmat):
        wmat = np.array(wmat.todense())

    znc = z0
    nwtn_stp, upd_fnorm, upd_fnorm_n = 0, None, None
    nwtn_upd_fnorms = []
    # import pdb
    # pdb.set_trace()

    while nwtn_stp < nwtn_adi_dict['nwtn_max_steps']:

        if znc is None:  # i.e., if z0 was None
            rhsadi = wmat
            mtxbt = None
        else:
            mtxb = mt * np.dot(znc, lau.mm_dnssps(znc.T, bmat))
            mtxbt = mtxb.T
            rhsadi = np.hstack([mtxb, wmat])
        # to avoid a dense matrix we use the smw formula
        # to compute (A-UV).-T
        # for the factorization mTxg.T =  tb * mTxtb.T = U*V
        # and we add the previous feedback:
        if mtxoldb is not None:
            mtxbt = mtxbt + mtxoldb.T

        znn = solve_proj_lyap_stein(amat=at, mmat=mt, jmat=jmat,
                                    wmat=rhsadi,
                                    umat=bmat, vmat=mtxbt,
                                    transposed=loctransposed,
                                    nwtn_adi_dict=nwtn_adi_dict)['zfac']

        if nwtn_adi_dict['full_upd_norm_check']:
            if znc is None:  # there was no initial guess
                znc = 0*znn
            upd_fnorm = lau.comp_sqfnrm_factrd_diff(znn, znc)
            upd_fnorm = np.sqrt(np.abs(upd_fnorm))

        else:
            if znc is None:  # there was no initial guess
                znc = 0*znn
            vec = np.random.randn(znn.shape[0], 1)
            vecn1 = comp_diff_zzv(znn, znc, vec)
            vec = np.random.randn(znn.shape[0], 1)
            vecn2 = comp_diff_zzv(znn, znc, vec)
            vec = np.random.randn(znn.shape[0], 1)
            # to make the estimate relative
            vecn3 = np.linalg.norm(np.dot(znn, np.dot(znn.T, vec)))
            if (vecn2 + vecn1)/vecn3 < 8e-9:
                upd_fnorm, nzn, nzc = lau.\
                    comp_sqfnrm_factrd_diff(znn, znc, ret_sing_norms=True)
                upd_fnorm_n = np.sqrt(np.abs(upd_fnorm) / np.abs(nzn))

        nwtn_upd_fnorms.append(upd_fnorm_n)
        try:
            if np.allclose(upd_fnorm_n, upd_fnorm):
                print 'no more change in the norm of the update... break'
                break
        except TypeError:
            pass
        if nwtn_adi_dict['full_upd_norm_check']:
            upd_fnorm = upd_fnorm_n
        elif (vecn2 + vecn1)/vecn3 < 8e-9:
            upd_fnorm = upd_fnorm_n

        try:
            if nwtn_adi_dict['verbose']:
                print ('Newton ADI step: {1} -- ' +
                       'rel f norm of update: {0}').format(upd_fnorm,
                                                           nwtn_stp + 1)
                if not nwtn_adi_dict['full_upd_norm_check']:
                    print ('btw, we decided whether to compute the actual ' +
                           'norm on the base of estimates:')
                    print '|| upd * vec || / || vec || = {0}'.format(vecn2)
                    print '||Z*vec|| = {0}'.format(vecn3)

        except KeyError:
            pass    # no verbosity specified - nothing is shown

        znc = znn
        nwtn_stp += 1
        if (upd_fnorm is not None
                and upd_fnorm < nwtn_adi_dict['nwtn_upd_reltol']):
            break

    return dict(zfac=znn, nwtn_upd_fnorms=nwtn_upd_fnorms)


def comp_proj_lyap_res_norm(Z, amat=None, mmat=None, wmat=None,
                            jmat=None, umat=None, vmat=None, Sinv=None):
    """compute the squared f norm of projected lyap residual

        res = Pt*[ Ft*ZZt*M + Mt*ZZt*M + W*Wt ]*P

    """
    if Z.shape[1] >= Z.shape[0]:
        raise Warning('TODO: catch cases where Z has more cols than rows')

    if Sinv is None:
        Mlu = spsla.factorized(mmat)
        MinvJt = lau.app_luinv_to_spmat(Mlu, jmat.T)
        Sinv = np.linalg.inv(jmat * MinvJt)

    def _app_pt(Z, jmat, MinvJt, Sinv):
        return Z - jmat.T * np.dot(Sinv, np.dot(MinvJt.T, Z))

    if umat is None and vmat is None:
        amattZ = amat.T * Z
    else:
        amattZ = amat.T*Z - lau.comp_uvz_spdns(vmat.T, umat.T, Z)

    PtFtZ = _app_pt(amattZ, jmat, MinvJt, Sinv)
    PtMtZ = _app_pt(mmat.T * Z, jmat, MinvJt, Sinv)
    PtW = _app_pt(wmat, jmat, MinvJt, Sinv)

    return lau.comp_sqfnrm_factrd_lyap_res(PtMtZ, PtFtZ, PtW)


def compress_Zsvd(Z, k=None, thresh=None, shplot=False, verbose=True):
    """routine that compresses the columns Z by means of a truncated SVD

    such that it ZZ.T is still well approximated"""

    nny = Z.shape[1]
    U, s, V = np.linalg.svd(Z, full_matrices=False)

    if shplot:
        import matplotlib.pyplot as plt
        plt.semilogy(s[:nny/2])
        plt.show(block=False)

    if k is None:
        k = nny

    if thresh is not None:
        k = min(k, np.where(s > thresh)[0].size)

    S = sps.dia_matrix((s[:k], 0), (k, k)).tocsr()

    if verbose:
        Zc = U[:, :k] * S
        # monitor the compression
        vec = np.random.randn(Z.shape[0], 1)
        print 'dims of Z and Z_rd: ', Z.shape, Zc.shape
        print '||(ZZ_rd - ZZ )*tstvec|| / ||ZZ_rd*tstvec|| = {0}'.\
            format(np.linalg.norm(np.dot(Z, np.dot(Z.T, vec)) -
                   np.dot(Zc, np.dot(Zc.T, vec))) /
                   np.linalg.norm(np.dot(Z, np.dot(Z.T, vec))))
        return Zc
    else:
        return U[:, :k] * S


def compress_ZQR(Z, kmax=None, shplot=False):
    """routine that compresses the columns Z by means of rank revealing QR

    such that it ZZ.T is still well approximated"""

    rmat, permumat = spla.qr(Z.T, mode='r', pivoting=True)

    if shplot:
        import matplotlib.pyplot as plt
        plt.show()

    return rmat[:kmax, np.argsort(permumat)].T


def comp_diff_zzv(zone, ztwo, vec):
    return np.linalg.norm(np.dot(zone, np.dot(zone.T, vec)) -
                          np.dot(ztwo, np.dot(ztwo.T, vec)))
