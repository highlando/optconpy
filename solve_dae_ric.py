import numpy as np
import dolfin_navier_scipy.data_output_utils as dou
import sadptprj_riclyap_adi.lin_alg_utils as lau
import sadptprj_riclyap_adi.proj_ric_utils as pru


def solve_flow_daeric(mmat=None, amat=None, jmat=None, bmat=None,
                      cmat=None, zzero=None, tmesh=None, tdatadict=None,
                      ystarvec=None, mycmat=None, rhsv=None, rhsp=None,
                      nwtn_adi_dict=None,
                      comprz_thresh=None, comprz_maxc=None, save_full_z=False,
                      get_tdpart=None, gttdprtargs=None,
                      get_datastr=None, gtdtstrargs=None):

    """

    Parameters
    ----------
    get_tdpart : callable f(t)
        returns the `mattd, rhstd` -- time dependent coefficients matrices
        and right hand side at time `t`
    gttdprtargs : dictionary
        `**kwargs` to get_tdpart

    """

    MT, AT, NV = mmat.T, amat.T, amat.shape[0]
    # set/compute the terminal values aka starting point
    # Zc = lau.apply_massinv(mmat, trct_mat)
    Zc = zzero

    cdatstr = get_datastr(time=tmesh[-1])

    mtxtb = pru.get_mTzzTtb(mmat.T, Zc, bmat)

    dou.save_npa(Zc, fstring=cdatstr + '__Z')
    dou.save_npa(mtxtb, fstring=cdatstr + '__mtxtb')

    if ystarvec is not None:
        wc = lau.apply_massinv(MT, np.dot(cmat, ystarvec(tmesh[-1])))
        dou.save_npa(wc, fstring=cdatstr + '__w')
    else:
        wc = None

    feedbackthroughdict = {tmesh[-1]: dict(w=cdatstr + '__w',
                                           mtxtb=cdatstr + '__mtxtb')}

    for tk, t in reversed(list(enumerate(tmesh[1:-1]))):
        cts = tmesh[tk+2] - t

        print 'Time is {0}, timestep is {1}'.\
            format(t, cts)

        # get the previous time time-dep matrices
        cdatstr = get_datastr(time=t)
        nmattd, rhsvtd = get_tdpart(time=t, **gttdprtargs)

        try:
            Zc = dou.load_npa(cdatstr + '__Z')
        except IOError:
            # coeffmat for nwtn adi
            ft_mat = -(0.5*MT + cts*(AT + nmattd.T))
            # rhs for nwtn adi
            w_mat = np.hstack([MT*Zc, np.sqrt(cts)*cmat])

            Zp = pru.proj_alg_ric_newtonadi(mmat=MT,
                                            amat=ft_mat, transposed=True,
                                            jmat=jmat,
                                            bmat=np.sqrt(cts)*bmat,
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

        ### and the affine correction
        at_mat = MT + cts*(AT + nmattd.T)

        # current rhs
        ftilde = rhsvtd + rhsv
        mtxft = pru.get_mTzzTtb(MT, Zc, ftilde)
        fl1 = mycmat.T * ystarvec(t)
        rhswc = MT*wc + cts*(fl1 - mtxft)

        mtxtb = pru.get_mTzzTtb(MT, Zc, bmat)

        wc = lau.solve_sadpnt_smw(amat=at_mat, jmat=jmat,
                                  umat=-cts*mtxtb, vmat=bmat.T,
                                  rhsv=rhswc)[:NV]

        dou.save_npa(wc, fstring=cdatstr + '__w')
        dou.save_npa(mtxtb, fstring=cdatstr + '__mtxtb')
        feedbackthroughdict.update({t: dict(w=cdatstr + '__w',
                                            mtxtb=cdatstr + '__mtxtb')})
    return feedbackthroughdict
