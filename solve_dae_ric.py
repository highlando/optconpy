import numpy as np
import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau
import sadptprj_riclyap_adi.proj_ric_utils as pru


def solve_flow_daeric(mmat=None, amat=None, jmat=None, bmat=None,
                      cmat=None, rhsv=None, rhsp=None,
                      mcmat=None, v_is_my=False,
                      rmat=None, vmat=None,
                      gamma=1.0,
                      tmesh=None, ystarvec=None,
                      nwtn_adi_dict=None,
                      curnwtnsdict=None,
                      comprz_thresh=None, comprz_maxc=None, save_full_z=False,
                      get_tdpart=None, gttdprtargs=None,
                      get_datastr=None, gtdtstrargs=None,
                      check_c_consist=True):

    """
    Routine for the solution of the DAE Riccati

    .. math::

        \\dot{MXM^T} + F^TXM + M^TXM + M^TXGXM + L(Y) = W \\\\
                JXM = 0 \\quad \\text{and} \\quad M^TXJ = 0 \\\\
                M^TX(T)M = W

    where :math:`F=A+N(t)`,
    where :math:`W:=C^T V C`, :math:`G:=B R^{-1} B^T`,
    and where :math:`L(Y)` is the Lagrange multiplier term.

    Simultaneously we solve for the feedforward term :math:`w`:

    .. math::

        \\dot{M^Tw} - [M^TXG+F^T]w - J^Tv = C^T V y^* + M^T[Xf + Yg] \\\\
                Jw = 0 \\\\
                M^Tw(T) = C^T V y^*(T)

    Note that :math:`V=M_y` if the norm of :math:`Y` is used
    in the cost function.


    Parameters
    ----------
    cmat : (NY, NV) array
        the (regularized aka projected) output matrix
    mcmat : (NY, NV) array
        output matrix times the mass matrix in the output space
    gamma : float, optional
        weighting parameter for penalization of the terminal value,
        TODO: rather provide the right weighting matrix V,
        defaults to `1.0`
    v_is_my : boolean
        whether the weighting matrix is the same as the mass matrix, \
                defaults to `False`
    get_tdpart : callable f(t)
        returns the `mattd, rhstd` -- time dependent coefficients matrices
        and right hand side at time `t`
    gtdtstrargs : dictionary
        **kwargs to the current data string
    gttdprtargs : dictionary
        `**kwargs` to get_tdpart

    Returns
    -------
    feedbackthroughdict : dictionary
        with time instances as keys and
        | `w` -- the current feedthrough value
        | `mtxtb` -- the current feedback gain part `(R.-1/2 * B.T * X * M).T`
        as values
    """

    if check_c_consist:
        if v_is_my and mcmat is not None:
            mic = lau.apply_massinv(mmat.T, mcmat.T)
            if np.linalg.norm(jmat*mic) > 1e-12:
                raise Warning('mcmat.T needs to be in the kernel of J*M.-1')
        elif cmat is not None:
            mic = lau.apply_massinv(mmat.T, cmat.T)
            if np.linalg.norm(jmat*mic) > 1e-12:
                raise Warning('cmat.T needs to be in the kernel of J*M.-1')

    MT, AT, NV = mmat.T, amat.T, amat.shape[0]

    gtdtstrargs.update(time=tmesh[-1])
    cdatstr = get_datastr(**gtdtstrargs)

    # set/compute the terminal values aka starting point
    if v_is_my and mcmat is not None:
        tct_mat = lau.apply_invsqrt_fromright(vmat, mcmat.T, output='dense')
    else:
        tct_mat = lau.apply_sqrt_fromright(vmat, cmat.T, output='dense')

    # TODO: good handling of bmat and umasmat
    tb_mat = lau.apply_invsqrt_fromright(rmat, bmat, output='sparse')
    # bmat_rpmo = bmat * np.linalg.inv(np.array(rmat.todense()))

    Zc = np.sqrt(gamma)*lau.apply_massinv(mmat, tct_mat)
    mtxtb = -pru.get_mTzzTtb(mmat.T, Zc, tb_mat)
    # mtxbrm = pru.get_mTzzTtb(mmat.T, Zc, bmat_rpmo)

    dou.save_npa(Zc, fstring=cdatstr + '__Z')
    dou.save_npa(mtxtb, fstring=cdatstr + '__mtxtb')

    if ystarvec is not None:
        wc = lau.apply_massinv(MT, gamma*np.dot(mcmat.T, ystarvec(tmesh[-1])))
        dou.save_npa(wc, fstring=cdatstr + '__w')
    else:
        wc = None

    feedbackthroughdict = {tmesh[-1]: dict(w=cdatstr + '__w',
                                           mtxtb=cdatstr + '__mtxtb')}

    # save the end values
    if curnwtnsdict is not None:
        dou.save_npa(wc, fstring=curnwtnsdict[tmesh[-1]]['w'])
        dou.save_npa(mtxtb, fstring=curnwtnsdict[tmesh[-1]]['mtxtb'])

    # time integration
    for tk, t in reversed(list(enumerate(tmesh[:-1]))):
        cts = tmesh[tk+1] - t

        print 'Time is {0}, timestep is {1}'.\
            format(t, cts)

        # get the previous time time-dep matrices
        gtdtstrargs.update(time=t)
        cdatstr = get_datastr(**gtdtstrargs)
        nmattd, rhsvtd = get_tdpart(time=t, **gttdprtargs)

        # get the feedback from the current newton step
        if curnwtnsdict is not None:
            try:
                cnsw = dou.load_npa(curnwtnsdict[t]['w'])
                cnsmtxtb = dou.load_npa(curnwtnsdict[t]['mtxtb'])
            except IOError:
                cnsw, cnsmtxtb = None, None
        else:
            cnsw, cnsmtxtb = None, None

        try:
            Zc = dou.load_npa(cdatstr + '__Z')
        except IOError:
            # coeffmat for nwtn adi
            ft_mat = -(0.5*MT + cts*(AT + nmattd.T))
            # rhs for nwtn adi
            w_mat = np.hstack([MT*Zc, np.sqrt(cts)*tct_mat])
            # feedback from a previous Newton step
            mtxb = np.sqrt(cts)*cnsmtxtb if cnsmtxtb is not None else None
            Zp = pru.proj_alg_ric_newtonadi(mmat=MT,
                                            amat=ft_mat, transposed=True,
                                            mtxoldb=mtxb,
                                            jmat=jmat,
                                            bmat=np.sqrt(cts)*tb_mat,
                                            wmat=w_mat, z0=Zc,
                                            nwtn_adi_dict=nwtn_adi_dict
                                            )['zfac']

            if comprz_maxc is not None or comprz_thresh is not None:
                Zc = pru.compress_Zsvd(Zp, thresh=comprz_thresh,
                                       k=comprz_maxc)
            else:
                Zc = Zp

            if save_full_z:
                dou.save_npa(Zp, fstring=cdatstr + '__Z')
            else:
                dou.save_npa(Zc, fstring=cdatstr + '__Z')

        # and the affine correction
        at_mat = MT + cts*(AT + nmattd.T)

        # current rhs
        ftilde = rhsvtd + rhsv

        # apply the feedback and through
        if cnsw is not None:
            ftilde = rhsvtd + rhsv + cnsw
        cnsmtxtb = cnsmtxtb + mtxtb if cnsmtxtb is not None else mtxtb

        mtxft = pru.get_mTzzTtb(MT, Zc, ftilde)

        fl1 = np.dot(mcmat.T, ystarvec(t))

        rhswc = MT*wc + cts*(fl1 - mtxft)

        mtxtb = -pru.get_mTzzTtb(MT, Zc, tb_mat)
        # mtxtbrm = pru.get_mTzzTtb(MT, Zc, bmat_rpmo)

        wc = lau.solve_sadpnt_smw(amat=at_mat, jmat=jmat,
                                  umat=cts*cnsmtxtb, vmat=tb_mat.T,
                                  rhsv=rhswc)[:NV]
        # wc = lau.solve_sadpnt_smw(amat=at_mat, jmat=jmat,
        #                           umat=-cts*mtxbrm, vmat=bmat.T,
        #                           rhsv=rhswc)[:NV]

        # update the feedback in Newton
        if curnwtnsdict is not None:
            cnsw = cnsw + wc if cnsw is not None else wc
            cnsmtxtb = cnsmtxtb + mtxtb if cnsmtxtb is not None else mtxtb
            dou.save_npa(cnsw, fstring=curnwtnsdict[t]['w'])
            dou.save_npa(cnsmtxtb, fstring=curnwtnsdict[t]['mtxtb'])

        dou.save_npa(wc, fstring=cdatstr + '__w')
        dou.save_npa(mtxtb, fstring=cdatstr + '__mtxtb')
        # dou.save_npa(mtxbrm, fstring=cdatstr + '__mtxbrm')
        feedbackthroughdict.update({t: dict(w=cdatstr + '__w',
                                            # mtxbrm=cdatstr + '__mtxbrm')})
                                            mtxtb=cdatstr + '__mtxtb')})

    return feedbackthroughdict
