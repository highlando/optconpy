import dolfin
import numpy as np
import os

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau
import sadptprj_riclyap_adi.proj_ric_utils as pru

import distr_control_fenics.cont_obs_utils as cou

dolfin.parameters.linear_algebra_backend = 'uBLAS'


def time_int_params(Nts):
    t0 = 0.0
    tE = 1.0
    dt = (tE - t0) / Nts
    sqzmesh = True,  # squeeze the mesh for shorter intervals towards the
                     # initial and terminal point, False for equidist
    tmesh = get_tint(t0, tE, Nts, sqzmesh)

    tip = dict(t0=t0,
               tE=tE,
               dt=dt,
               Nts=Nts,
               tmesh=tmesh,
               vfile=None,
               pfile=None,
               Residuals=[],
               ParaviewOutput=True,
               proutdir='results/',
               prfprfx='',
               nnewtsteps=9,  # n nwtn stps for vel comp, 0 for Stokes flow
               vel_nwtn_tol=1e-14,
               norm_nwtnupd_list=[],
               # parameters for newton adi iteration
               nwtn_adi_dict=dict(
                   adi_max_steps=100,
                   adi_newZ_reltol=1e-5,
                   nwtn_max_steps=7,
                   nwtn_upd_reltol=4e-8,
                   nwtn_upd_abstol=1e-7,
                   verbose=True,
                   full_upd_norm_check=False,
                   check_lyap_res=False
               ),
               compress_z=True,  # whether or not to compress Z
               comprz_maxc=500,  # compression of the columns of Z by QR
               comprz_thresh=5e-5,  # threshold for trunc of SVD
               save_full_z=False,  # whether or not to save the uncompressed Z
               yscomp=[],
               ystar=[]
               )

    return tip


def set_vpfiles(tip, fstring='not specified'):
    tip['pfile'] = dolfin.File(fstring+'_p.pvd')
    tip['vfile'] = dolfin.File(fstring+'_vel.pvd')


def get_tint(t0, tE, Nts, sqzmesh):
    """set up the time mesh """
    if sqzmesh:
        taux = np.linspace(-0.5*np.pi, 0.5*np.pi, Nts+1)
        taux = (np.sin(taux) + 1)*0.5  # squeeze and adjust to [0, 1]
        tint = (t0 + (tE-t0)*taux).flatten().tolist()  # adjust to [t0, tE]
    else:
        tint = np.linspace(t0, tE, Nts+1).flatten().tolist()

    return tint


def get_datastr(nwtn=None, time=None,
                meshp=None, nu=None, Nts=None, dt=None,
                data_prfx=''):

    return (data_prfx +
            'Nwtnit{0}_time{1}_nu{2}_mesh{3}_Nts{4}_dt{5}').format(
        nwtn, time, nu, meshp, Nts, dt)


def optcon_nse(problemname='drivencavity',
               N=10, Nts=10, nu=1e-2):

    tip = time_int_params(Nts)

    problemdict = dict(drivencavity=dnsps.drivcav_fems,
                       cylinderwake=dnsps.cyl_fems)

    problemfem = problemdict[problemname]
    femp = problemfem(N)

    data_prfx = problemname + '__'
    NU, NY = 3, 4

    # specify in what spatial direction Bu changes. The remaining is constant
    if problemname == 'drivencavity':
        uspacedep = 0
    elif problemname == 'cylinderwake':
        uspacedep = 1

    # output
    ddir = 'data/'
    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

    stokesmats = dts.get_stokessysmats(femp['V'], femp['Q'],
                                       tip['nu'])

    rhsd_vf = dts.setget_rhs(femp['V'], femp['Q'],
                             femp['fv'], femp['fp'], t=0)

    # remove the freedom in the pressure
    stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
    stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]
    rhsd_vf['fp'] = rhsd_vf['fp'][:-1, :]

    # reduce the matrices by resolving the BCs
    (stokesmatsc,
     rhsd_stbc,
     invinds,
     bcinds,
     bcvals) = dts.condense_sysmatsbybcs(stokesmats,
                                         femp['diribcs'])

    # pressure freedom and dirichlet reduced rhs
    rhsd_vfrc = dict(fpr=rhsd_vf['fp'], fvc=rhsd_vf['fv'][invinds, ])

    # add the info on boundary and inner nodes
    bcdata = {'bcinds': bcinds,
              'bcvals': bcvals,
              'invinds': invinds}
    femp.update(bcdata)

    # casting some parameters
    NV, DT, INVINDS = len(femp['invinds']), tip['dt'], femp['invinds']

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(fv_stbc=rhsd_stbc['fv'], fp_stbc=rhsd_stbc['fp'],
                   N=N, nu=tip['nu'],
                   nnewtsteps=tip['nnewtsteps'],
                   vel_nwtn_tol=tip['vel_nwtn_tol'],
                   ddir=ddir, get_datastring=get_datastr,
                   data_prfx=data_prfx,
                   paraviewoutput=tip['ParaviewOutput'],
                   vfileprfx=tip['proutdir']+'vel_',
                   pfileprfx=tip['proutdir']+'p_')

    contp = cou.ContParams(femp['V'])
#
# compute the uncontrolled steady state Navier-Stokes solution
#
    # v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)
    newtk = snu.solve_nse(return_nwtn_step=True, **soldict)

#
# Prepare for control
#

    # casting some parameters
    NY, NU = contp.NY, contp.NU

    contsetupstr = problemname + '__NV{0}NU{1}NY{2}'.format(NV, NU, NY)

    # get the control and observation operators
    try:
        b_mat = dou.load_spa(ddir + contsetupstr + '__b_mat')
        u_masmat = dou.load_spa(ddir + contsetupstr + '__u_masmat')
        print 'loaded `b_mat`'
    except IOError:
        print 'computing `b_mat`...'
        b_mat, u_masmat = cou.get_inp_opa(cdcoo=femp['cdcoo'], V=femp['V'],
                                          NU=NU, xcomp=uspacedep)
        dou.save_spa(b_mat, ddir + contsetupstr + '__b_mat')
        dou.save_spa(u_masmat, ddir + contsetupstr + '__u_masmat')
    try:
        mc_mat = dou.load_spa(ddir + contsetupstr + '__mc_mat')
        y_masmat = dou.load_spa(ddir + contsetupstr + '__y_masmat')
        print 'loaded `c_mat`'
    except IOError:
        print 'computing `c_mat`...'
        mc_mat, y_masmat = cou.get_mout_opa(odcoo=femp['odcoo'],
                                            V=femp['V'], NY=NY)
        dou.save_spa(mc_mat, ddir + contsetupstr + '__mc_mat')
        dou.save_spa(y_masmat, ddir + contsetupstr + '__y_masmat')

    # restrict the operators to the inner nodes
    mc_mat = mc_mat[:, invinds][:, :]
    b_mat = b_mat[invinds, :][:, :]
    contsetupstr = 'NV{0}NU{1}NY{2}'.format(NV, NU, NY)

    # for further use:
    c_mat = lau.apply_massinv(y_masmat, mc_mat)
    mct_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                         jmat=stokesmatsc['J'],
                                         rhsv=mc_mat.T,
                                         transposedprj=True)

    # set the weighing matrices
    # if contp.R is None:
    contp.R = contp.alphau * u_masmat
    # TODO: by now we tacitly assume that V, W = MyC.T My^-1 MyC
    # if contp.V is None:
    #     contp.V = My
    # if contp.W is None:
    #     contp.W = My

#
# solve the differential-alg. Riccati eqn for the feedback gain X
# via computing factors Z, such that X = -Z*Z.T
#
# at the same time we solve for the affine-linear correction w
#

    # tilde B = BR^{-1/2}
    tb_mat = lau.apply_invsqrt_fromleft(contp.R, b_mat,
                                        output='sparse')
    tb_dense = np.array(tb_mat.todense())

    trct_mat = lau.apply_invsqrt_fromleft(y_masmat,
                                          mct_mat_reg, output='dense')

    cntpstr = 'NV{3}NY{0}NU{1}alphau{2}'.format(contp.NU, contp.NY,
                                                contp.alphau, NV)

    # set/compute the terminal values aka starting point
    Zc = lau.apply_massinv(stokesmatsc['M'], trct_mat)
    wc = lau.apply_massinv(stokesmatsc['MT'],
                           np.dot(mct_mat_reg, contp.ystarvec(tip['tE'])))

    cdatstr = get_datastr(nwtn=newtk, time=tip['tE'], meshp=N,
                          Nts=Nts, data_prfx=data_prfx)

    dou.save_npa(Zc, fstring=ddir + cdatstr + cntpstr + '__Z')
    dou.save_npa(wc, fstring=ddir + cdatstr + cntpstr + '__w')

    # we gonna use this quite often
    MT, AT = stokesmatsc['MT'], stokesmatsc['AT']
    M, A = stokesmatsc['M'], stokesmatsc['A']

    # time_before_soldaeric = time.time()
    for tk, t in reversed(list(enumerate(tip['tmesh'][:-1]))):
    # for t in np.linspace(tip['tE'] -  DT, tip['t0'], Nts):
        cts = tip['tmesh'][tk+1] - t

        print 'Time is {0}, DT is {1}'.format(t, cts)

        # get the previous time convection matrices
        pdatstr = get_datastr(nwtn=newtk, time=t, meshp=N,
                              Nts=Nts, data_prfx=data_prfx)
        prev_v = dou.load_npa(ddir + pdatstr + '__vel')
        (convc_mat, rhs_con,
         rhsv_conbc) = snu.get_v_conv_conts(prev_v=prev_v, invinds=invinds,
                                            V=femp['V'],
                                            diribcs=femp['diribcs'])

        try:
            Zc = dou.load_npa(ddir + pdatstr + cntpstr + '__Z')
        except IOError:

            # coeffmat for nwtn adi
            ft_mat = -(0.5*stokesmatsc['MT'] + cts*(stokesmatsc['AT'] +
                                                    convc_mat.T))
            # rhs for nwtn adi
            w_mat = np.hstack([stokesmatsc['MT']*Zc, np.sqrt(cts)*trct_mat])

            Zp = pru.proj_alg_ric_newtonadi(mmat=stokesmatsc['MT'],
                                            fmat=ft_mat, transposed=True,
                                            jmat=stokesmatsc['J'],
                                            bmat=np.sqrt(cts)*tb_mat,
                                            wmat=w_mat, z0=Zc,
                                            nwtn_adi_dict=tip['nwtn_adi_dict']
                                            )['zfac']

            if tip['compress_z']:
                # Zc = pru.compress_ZQR(Zp, kmax=tip['comprz_maxc'])
                Zc = pru.compress_Zsvd(Zp, thresh=tip['comprz_thresh'])
                # monitor the compression
                vec = np.random.randn(Zp.shape[0], 1)
                print 'dims of Z and Z_red: ', Zp.shape, Zc.shape
                print '||(ZZ_red - ZZ )*testvec|| / ||ZZ_red*testvec|| = {0}'.\
                    format(np.linalg.norm(np.dot(Zp, np.dot(Zp.T, vec)) -
                           np.dot(Zc, np.dot(Zc.T, vec))) /
                           np.linalg.norm(np.dot(Zp, np.dot(Zp.T, vec))))
            else:
                Zc = Zp

            if tip['save_full_z']:
                dou.save_npa(Zp, fstring=ddir + pdatstr + cntpstr + '__Z')
            else:
                dou.save_npa(Zc, fstring=ddir + pdatstr + cntpstr + '__Z')

        ### and the affine correction
        at_mat = MT + cts*(AT + convc_mat.T)

        # current rhs
        ftilde = rhs_con[INVINDS, :] + rhsv_conbc + rhsd_vfstbc['fv']
        mtxft = pru.get_mTzzTtb(M.T, Zc, ftilde)
        fl1 = mc_mat.T * contp.ystarvec(t)
        rhswc = MT*wc + cts*(fl1 - mtxft)

        mtxtb = pru.get_mTzzTtb(M.T, Zc, tb_mat)

        wc = lau.solve_sadpnt_smw(amat=at_mat, jmat=stokesmatsc['J'],
                                  umat=-cts*mtxtb, vmat=tb_mat.T,
                                  rhsv=rhswc)[:NV]

        dou.save_npa(wc, fstring=ddir + pdatstr + cntpstr + '__w')

    time_after_soldaeric = time.time()

    # solve the closed loop system
    set_vpfiles(tip, fstring=('results/' + 'closedloop' + cntpstr +
                              'NewtonIt{0}').format(newtk))

    v_old = inivalvec
    yn = np.dot(c_mat, v_old)
    tip['yscomp'].append(yn.flatten().tolist())
    tip['ystar'].append(contp.ystarvec(0).flatten().tolist())

    for tk, t in enumerate(tip['tmesh'][1:]):
        cts = t - tip['tmesh'][tk]

        print 'Time is {0}, DT is {1}'.format(t, cts)

        # t for implicit scheme
        ndatstr = get_datastr(nwtn=newtk, time=t,
                              meshp=N, timps=tip)

        # convec mats
        next_v = dou.load_npa(ddir + ndatstr + '__vel')
        convc_mat, rhs_con, rhsv_conbc = get_v_conv_conts(next_v,
                                                          femp, tip)

        # feedback mat and feedthrough
        next_zmat = dou.load_npa(ddir + ndatstr + cntpstr + '__Z')
        next_w = dou.load_npa(ddir + ndatstr + cntpstr + '__w')

        # rhs
        fvn = rhs_con[INVINDS, :] + rhsv_conbc + rhsd_vfstbc['fv']
        rhsn = M*next_v + cts*(fvn + tb_mat * (tb_mat.T * next_w))

        # coeffmats
        amat = M + cts*(A + convc_mat)
        mtxtb = pru.get_mTzzTtb(M.T, next_zmat, tb_mat)

        # TODO: rhsp!!!
        vpn = lau.solve_sadpnt_smw(amat=amat, jmat=stokesmatsc['J'],
                                   rhsv=rhsn,
                                   umat=-cts*tb_dense, vmat=mtxtb.T)

        # vpn = np.atleast_2d(sps.linalg.spsolve(amat, currhs)).T
        v_old = vpn[:NV]

        yn = np.dot(c_mat, vpn[:NV])
        print 'norm of current w: ', np.linalg.norm(next_w)
        print 'current y: ', yn

        tip['yscomp'].append(yn.flatten().tolist())
        tip['ystar'].append(contp.ystarvec(0).flatten().tolist())

        dou.save_npa(vpn[:NV], fstring=ddir + cdatstr + '__cont_vel')

        dou.output_paraview(tip, femp, vp=vpn, t=t),

    dou.save_output_json(tip['yscomp'], tip['tmesh'], ystar=tip['ystar'],
                         fstring=ddir + cdatstr + cntpstr + '__sigout')

    print 'dim of v :', femp['V'].dim()
    print 'time for solving dae ric :', \
        time_after_soldaeric - time_before_soldaeric

if __name__ == '__main__':
    optcon_nse(N=25, Nts=40)
