import dolfin
import numpy as np
import scipy.sparse as sps
# import matplotlib.pyplot as plt
import os
import glob

import dolfin_to_nparrays as dtn
import lin_alg_utils as lau
import data_output_utils as dou
import cont_obs_utils as cou
import proj_ric_utils as pru

dolfin.parameters.linear_algebra_backend = 'uBLAS'


def time_int_params(Nts):
    t0 = 0.0
    tE = 1.0
    dt = (tE - t0) / Nts
    tip = dict(t0=t0,
               tE=tE,
               dt=dt,
               Nts=Nts,
               Navier=True,  # set 0 for Stokes flow and 1 for NS
               vfile=None,
               pfile=None,
               Residuals=[],
               ParaviewOutput=True,
               nu=1e-2,
               nnewtsteps=4,  # n nwtn stps for vel comp
               vel_nwtn_tol=1e-14,
               norm_nwtnupd_list=[],
               # parameters for newton adi iteration
               nwtn_adi_dict=dict(
                   adi_max_steps=100,
                   adi_newZ_reltol=1e-5,
                   nwtn_max_steps=6,
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
               )

    return tip


def set_vpfiles(tip, fstring='not specified'):
    tip['pfile'] = dolfin.File(fstring+'_p.pvd')
    tip['vfile'] = dolfin.File(fstring+'_vel.pvd')


class ContParams():
    """define the parameters of the control problem

    as there are
    - dimensions of in and output space
    - extensions of the subdomains of control and observation
    - weighting matrices (if None, then massmatrix)
    - desired output
    """
    def __init__(self):

        self.ystarx = dolfin.Expression('0.0', t=0)
        self.ystary = dolfin.Expression('1.0', t=0)
        # if t, then add t=0 to both comps !!1!!11

        self.NU, self.NY = 4, 4

        self.odcoo = dict(xmin=0.45,
                          xmax=0.55,
                          ymin=0.5,
                          ymax=0.7)
        self.cdcoo = dict(xmin=0.4,
                          xmax=0.6,
                          ymin=0.2,
                          ymax=0.3)

        self.R = None
        # regularization parameter
        self.alphau = 1e-7
        self.endpy = 10
        self.V = None
        self.W = None

        self.ymesh = dolfin.IntervalMesh(self.NY-1, self.odcoo['ymin'],
                                         self.odcoo['ymax'])
        self.Y = dolfin.FunctionSpace(self.ymesh, 'CG', 1)
        # TODO: pass Y to cou.get_output_operator

    def ystarvec(self, t=None):
        """return the current value of ystar

        as np array [ystar1
                     ystar2] """
        if t is None:
            try:
                self.ystarx.t, self.ystary.t = t, t
            except AttributeError:
                pass  # everything's cool - ystar does not dep on t
            else:
                raise Warning('You need provide a time for ystar')
        else:
            try:
                self.ystarx.t, self.ystary.t = t, t
            except AttributeError:
                raise UserWarning('no time dependency of ystar' +
                                  'the provided t is ignored')

        ysx = dolfin.interpolate(self.ystarx, self.Y)
        ysy = dolfin.interpolate(self.ystary, self.Y)
        return np.vstack([np.atleast_2d(ysx.vector().array()).T,
                          np.atleast_2d(ysy.vector().array()).T])


def get_datastr(nwtn=None, time=None, meshp=None, timps=None):

    if timps['Navier']:
        navsto = 'NStokes'
    else:
        navsto = 'Stokes'

    return (navsto + 'Nwtnit{0}_time{1}_nu{2}_mesh{3}_Nts{4}_dt{5}').format(
        nwtn, time, timps['nu'], meshp,
        timps['Nts'], timps['dt']
    )


def drivcav_fems(N, NU=None, NY=None):
    """dictionary for the fem items of the (unit) driven cavity

    """
    mesh = dolfin.UnitSquareMesh(N, N)
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    Q = dolfin.FunctionSpace(mesh, "CG", 1)
    # pressure node that is set to zero

    # Boundaries
    def top(x, on_boundary):
        return x[1] > 1.0 - dolfin.DOLFIN_EPS

    def leftbotright(x, on_boundary):
        return (x[0] > 1.0 - dolfin.DOLFIN_EPS
                or x[1] < dolfin.DOLFIN_EPS
                or x[0] < dolfin.DOLFIN_EPS)

    # No-slip boundary condition for velocity
    noslip = dolfin.Constant((0.0, 0.0))
    bc0 = dolfin.DirichletBC(V, noslip, leftbotright)
    # Boundary condition for velocity at the lid
    lid = dolfin.Constant(("1", "0.0"))
    bc1 = dolfin.DirichletBC(V, lid, top)
    # Collect boundary conditions
    diribcs = [bc0, bc1]
    # rhs of momentum eqn
    fv = dolfin.Constant((0.0, 0.0))
    # rhs of the continuity eqn
    fp = dolfin.Constant(0.0)

    dfems = dict(mesh=mesh,
                 V=V,
                 Q=Q,
                 diribcs=diribcs,
                 fv=fv,
                 fp=fp)

    return dfems


def get_v_conv_conts(prev_v, femp, tip):

    # get and condense the linearized convection
    # rhsv_con += (u_0*D_x)u_0 from the Newton scheme
    if tip['Navier']:
        N1, N2, rhs_con = dtn.get_convmats(u0_vec=prev_v,
                                           V=femp['V'],
                                           invinds=femp['invinds'],
                                           diribcs=femp['diribcs'])
        convc_mat, rhsv_conbc = \
            dtn.condense_velmatsbybcs(N1 + N2, femp['diribcs'])

    else:
        nnvv = femp['invinds'].size
        convc_mat, rhsv_conbc = sps.csr_matrix((nnvv, nnvv)), 0
        rhs_con = np.zeros((femp['V'].dim(), 1))

    return convc_mat, rhs_con, rhsv_conbc


def setup_sadpnt_matsrhs(amat, jmat, rhsv, rhsp=None, jmatT=None):

    nnpp = jmat.shape[0]

    if jmatT is None:
        jmatT = jmat.T
    if rhsp is None:
        rhsp = np.zeros((nnpp, 1))

    sysm1 = sps.hstack([amat, jmat.T], format='csr')
    sysm2 = sps.hstack([jmat, sps.csr_matrix((nnpp, nnpp))], format='csr')

    mata = sps.vstack([sysm1, sysm2], format='csr')
    rhs = np.vstack([rhsv, rhsp])

    return mata, rhs


def optcon_nse(N=10, Nts=10):

    tip = time_int_params(Nts)
    femp = drivcav_fems(N)
    contp = ContParams()

    # output
    ddir = 'data/'
    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

    if tip['ParaviewOutput']:
        os.chdir('results/')
        for fname in glob.glob('NewtonIt' + '*'):
            os.remove(fname)
        os.chdir('..')

#
# start with the Stokes problem for initialization
#

    stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'],
                                       tip['nu'])

    rhsd_vf = dtn.setget_rhs(femp['V'], femp['Q'],
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
     bcvals) = dtn.condense_sysmatsbybcs(stokesmats,
                                         femp['diribcs'])

    # we will need transposes, and explicit is better than implicit
    # here, the coefficient matrices are symmetric
    stokesmatsc.update(dict(MT=stokesmatsc['M'],
                            AT=stokesmatsc['A']))

    # add the info on boundary and inner nodes
    bcdata = {'bcinds': bcinds,
              'bcvals': bcvals,
              'invinds': invinds}
    femp.update(bcdata)

    # casting some parameters
    NV, DT, INVINDS = len(femp['invinds']), tip['dt'], femp['invinds']
    NP = stokesmatsc['J'].shape[0]
    # and setting current values
    newtk, t = 0, None

    # compute the steady state stokes solution
    rhsd_vfstbc = dict(fv=rhsd_stbc['fv'] +
                       rhsd_vf['fv'][INVINDS, ],
                       fp=rhsd_stbc['fp'] + rhsd_vf['fp'])

    vp_stokes = lau.stokes_steadystate(matdict=stokesmatsc,
                                       rhsdict=rhsd_vfstbc)

    # save the data
    cdatstr = get_datastr(nwtn=newtk, time=t,
                          meshp=N, timps=tip)
    dou.save_npa(vp_stokes[:NV, ], fstring=ddir + cdatstr + '__vel')

#
# Compute the time-dependent flow
#

    # Stokes solution as initial value
    inivalvec = vp_stokes[:NV, ]

    norm_nwtnupd = 1
    while newtk < tip['nnewtsteps']:
        newtk += 1
        # check for previously computed velocities
        try:
            cdatstr = get_datastr(nwtn=newtk, time=tip['tE'],
                                  meshp=N, timps=tip)

            norm_nwtnupd = dou.load_npa(ddir + cdatstr + '__norm_nwtnupd')
            prev_v = dou.load_npa(ddir + cdatstr + '__vel')

            tip['norm_nwtnupd_list'].append(norm_nwtnupd)
            print 'found vel files of Newton iteration {0}'.format(newtk)
            print 'norm of current Nwtn update: {0}'.format(norm_nwtnupd[0])

        except IOError:
            newtk -= 1
            break

    while (newtk < tip['nnewtsteps'] and
           norm_nwtnupd > tip['vel_nwtn_tol']):
        newtk += 1

        cdatstr = get_datastr(nwtn=newtk, time=tip['t0'],
                              meshp=N, timps=tip)

        # save the inival value
        dou.save_npa(inivalvec, fstring=ddir + cdatstr + '__vel')

        set_vpfiles(tip, fstring=('results/' +
                                  'NewtonIt{0}').format(newtk))
        dou.output_paraview(tip, femp, vp=vp_stokes, t=0)

        norm_nwtnupd = 0
        v_old = inivalvec  # start vector in every Newtonit
        print 'Computing Newton Iteration {0} -- ({1} timesteps)'.\
            format(newtk, Nts)

        for t in np.linspace(tip['t0']+DT, tip['tE'], Nts):
            cdatstr = get_datastr(nwtn=newtk, time=t,
                                  meshp=N, timps=tip)

            # t for implicit scheme
            pdatstr = get_datastr(nwtn=newtk-1, time=t,
                                  meshp=N, timps=tip)

            # try - except for linearizations about stationary sols
            # for which t=None
            try:
                prev_v = dou.load_npa(ddir + pdatstr + '__vel')
            except IOError:
                pdatstr = get_datastr(nwtn=newtk - 1, time=None,
                                      meshp=N, timps=tip)
                prev_v = dou.load_npa(ddir + pdatstr + '__vel')

            convc_mat, rhs_con, rhsv_conbc = get_v_conv_conts(prev_v,
                                                              femp, tip)

            rhsd_cur = dict(fv=stokesmatsc['M'] * v_old +
                            DT * (rhs_con[INVINDS, :] +
                                  rhsv_conbc + rhsd_vfstbc['fv']),
                            fp=rhsd_vfstbc['fp'])

            matd_cur = dict(A=stokesmatsc['M'] +
                            DT * (stokesmatsc['A'] + convc_mat),
                            JT=stokesmatsc['JT'],
                            J=stokesmatsc['J'])

            vp = lau.stokes_steadystate(matdict=matd_cur,
                                        rhsdict=rhsd_cur)

            v_old = vp[:NV, ]

            dou.save_npa(v_old, fstring=ddir + cdatstr + '__vel')

            dou.output_paraview(tip, femp, vp=vp, t=t),

            # integrate the Newton error
            norm_nwtnupd += DT * np.dot((v_old - prev_v).T,
                                        stokesmatsc['M'] *
                                        (v_old - prev_v))

        dou.save_npa(norm_nwtnupd, ddir + cdatstr + '__norm_nwtnupd')
        tip['norm_nwtnupd_list'].append(norm_nwtnupd[0])

        print 'norm of current Newton update: {}'.format(norm_nwtnupd)

#
# Prepare for control
#

    # casting some parameters
    NY, NU = contp.NY, contp.NU

    contsetupstr = 'NV{0}NU{1}NY{2}'.format(NV, NU, NY)

    # get the control and observation operators
    try:
        b_mat = dou.load_spa(ddir + contsetupstr + '__b_mat')
        u_masmat = dou.load_spa(ddir + contsetupstr + '__u_masmat')
        print 'loaded `b_mat`'
    except IOError:
        print 'computing `b_mat`...'
        b_mat, u_masmat = cou.get_inp_opa(cdcoo=contp.cdcoo,
                                          V=femp['V'], NU=contp.NU)
        dou.save_spa(b_mat, ddir + contsetupstr + '__b_mat')
        dou.save_spa(u_masmat, ddir + contsetupstr + '__u_masmat')
    try:
        mc_mat = dou.load_spa(ddir + contsetupstr + '__mc_mat')
        y_masmat = dou.load_spa(ddir + contsetupstr + '__y_masmat')
        print 'loaded `c_mat`'
    except IOError:
        print 'computing `c_mat`...'
        mc_mat, y_masmat = cou.get_mout_opa(odcoo=contp.odcoo,
                                            V=femp['V'], NY=contp.NY)
        dou.save_spa(mc_mat, ddir + contsetupstr + '__mc_mat')
        dou.save_spa(y_masmat, ddir + contsetupstr + '__y_masmat')

    # restrict the operators to the inner nodes
    mc_mat = mc_mat[:, invinds][:, :]
    b_mat = b_mat[invinds, :][:, :]

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

    trct_mat = lau.apply_invsqrt_fromleft(contp.endpy*y_masmat,
                                          mct_mat_reg, output='dense')

    cntpstr = 'NY{0}NU{1}alphau{2}'.format(contp.NU, contp.NY, contp.alphau)

    # set/compute the terminal values aka starting point
    Zc = lau.apply_massinv(stokesmatsc['M'], trct_mat)
    wc = -lau.apply_massinv(stokesmatsc['MT'],
                            np.dot(mct_mat_reg, contp.ystarvec(tip['tE'])))

    cdatstr = get_datastr(nwtn=newtk, time=tip['tE'], meshp=N, timps=tip)

    dou.save_npa(Zc, fstring=ddir + cdatstr + cntpstr + '__Z')
    dou.save_npa(wc, fstring=ddir + cdatstr + cntpstr + '__w')

    # we gonna use this quite often
    MT, AT = stokesmatsc['MT'], stokesmatsc['AT']
    M, A = stokesmatsc['M'], stokesmatsc['A']

    for t in np.linspace(tip['tE'] - DT, tip['t0'], Nts):
        print 'Time is {0}'.format(t)

        # get the previous time convection matrices
        pdatstr = get_datastr(nwtn=newtk, time=t, meshp=N, timps=tip)
        prev_v = dou.load_npa(ddir + pdatstr + '__vel')
        convc_mat, rhs_con, rhsv_conbc = get_v_conv_conts(prev_v,
                                                          femp, tip)

        try:
            Zc = dou.load_npa(ddir + pdatstr + cntpstr + '__Z')
        except IOError:

            # coeffmat for nwtn adi
            ft_mat = -(0.5*stokesmatsc['MT'] + DT*(stokesmatsc['AT'] +
                                                   convc_mat.T))
            # rhs for nwtn adi
            w_mat = np.hstack([stokesmatsc['MT']*Zc, np.sqrt(DT)*trct_mat])

            Zp = pru.proj_alg_ric_newtonadi(mmat=stokesmatsc['MT'],
                                            fmat=ft_mat, transposed=True,
                                            jmat=stokesmatsc['J'],
                                            bmat=np.sqrt(DT)*tb_mat,
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
        ftilde = rhs_con[INVINDS, :] + rhsv_conbc + rhsd_vfstbc['fv']
        at_mat = MT + DT*(AT + convc_mat.T)
        rhswc = MT*wc + DT*(mc_mat.T*contp.ystarvec(t) -
                            MT*np.dot(Zc, np.dot(Zc.T, ftilde)))

        amat, currhs = setup_sadpnt_matsrhs(at_mat, stokesmatsc['J'], rhswc)

        umat = DT*MT*np.dot(Zc, Zc.T*tb_mat)
        vmat = tb_mat.T

        vmate = sps.hstack([vmat, sps.csc_matrix((vmat.shape[0], NP))])
        umate = np.vstack([umat, np.zeros((NP, umat.shape[1]))])

        wc = lau.app_smw_inv(amat, umat=-umate, vmat=vmate, rhsa=currhs)[:NV]
        dou.save_npa(wc, fstring=ddir + pdatstr + cntpstr + '__w')

    # solve the closed loop system
    set_vpfiles(tip, fstring=('results/' + 'closedloop' + cntpstr +
                              'NewtonIt{0}').format(newtk))

    v_old = inivalvec
    for t in np.linspace(tip['t0']+DT, tip['tE'], Nts):

        # t for implicit scheme
        ndatstr = get_datastr(nwtn=newtk, time=t,
                              meshp=N, timps=tip)

        # convec mats
        next_v = dou.load_npa(ddir + ndatstr + '__vel')
        convc_mat, rhs_con, rhsv_conbc = get_v_conv_conts(next_v,
                                                          femp, tip)

        # feedback mats
        next_zmat = dou.load_npa(ddir + ndatstr + cntpstr + '__Z')
        next_w = dou.load_npa(ddir + ndatstr + cntpstr + '__w')
        print 'norm of w:', np.linalg.norm(next_w)

        umat = DT*MT*np.dot(next_zmat, next_zmat.T*tb_mat)
        vmat = tb_mat.T

        vmate = sps.hstack([vmat, sps.csc_matrix((vmat.shape[0], NP))])
        umate = DT*np.vstack([umat, np.zeros((NP, umat.shape[1]))])

        fvn = rhs_con[INVINDS, :] + rhsv_conbc + rhsd_vfstbc['fv']
        # rhsn = M*next_v + DT*(fvn + tb_mat * (tb_mat.T * next_w))
        rhsn = M*v_old + DT*(fvn + 0*tb_mat * (tb_mat.T * next_w))

        amat = M + DT*(A + convc_mat)
        rvec = np.random.randn(next_zmat.shape[0], 1)
        print 'norm of amat', np.linalg.norm(amat*rvec)
        print 'norm of gain mat', np.linalg.norm(np.dot(umat, vmat*rvec))

        amat, currhs = setup_sadpnt_matsrhs(amat, stokesmatsc['J'], rhsn)

        vpn = lau.app_smw_inv(amat, umat=-umate, vmat=vmate, rhsa=currhs)
        # vpn = np.atleast_2d(sps.linalg.spsolve(amat, currhs)).T
        v_old = vpn[:NV]

        yn = lau.apply_massinv(y_masmat, mc_mat*vpn[:NV])
        print 'current y: ', yn

        dou.save_npa(vpn[:NV], fstring=ddir + cdatstr + '__cont_vel')

        dou.output_paraview(tip, femp, vp=vpn, t=t),

    print 'dim of v :', femp['V'].dim()

if __name__ == '__main__':
    optcon_nse(N=15, Nts=2)
