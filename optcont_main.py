import dolfin
import numpy as np
# import scipy.sparse as sps
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
               vfile=None,
               pfile=None,
               Residuals=[],
               ParaviewOutput=True,
               nu=1e-3,
               nnewtsteps=3,  # n nwtn stps for vel comp
               vel_nwtn_tol=1e-14,
               norm_nwtnupd_list=[],
               # parameters for newton adi iteration
               nwtn_adi_dict=dict(
                   adi_max_steps=120,
                   adi_newZ_reltol=1e-8,
                   nwtn_max_steps=24,
                   nwtn_upd_reltol=4e-8,
                   nwtn_upd_abstol=4e-8,
                   verbose=True
               )
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

        ystar1 = dolfin.Expression('1', t=0)
        ystar2 = dolfin.Expression('1', t=0)

        self.ystar = [ystar1, ystar2]
        self.NU, self.NY = 10, 7

        self.odcoo = dict(xmin=0.45,
                          xmax=0.55,
                          ymin=0.5,
                          ymax=0.7)
        self.cdcoo = dict(xmin=0.4,
                          xmax=0.6,
                          ymin=0.2,
                          ymax=0.3)

        self.R = None,
        # regularization parameter
        self.alphau = 1e-4
        self.V = None
        self.W = None

    def ystarvec(self, t=None):
        """return the current value of ystar

        as np array [ystar1
                     ystar2] """
        if t is not None:
            try:
                self.ystar1.t, self.ystar2.t = t, t
                raise Warning('You need provide a time for ystar')
            except AttributeError:
                pass  # everything's cool - ystar does not dep on t
        else:
            try:
                self.ystar1.t, self.ystar2.t = t, t
            except AttributeError:
                raise UserWarning('no time dependency of ystar' +
                                  'the provided t is ignored')

        return np.vstack([self.ystar1.vector(),
                          self.ystar2.vector()])


def get_datastr(nwtn=None, time=None, meshp=None, timps=None):
    return 'Nwit{0}Time{1}Nu{2}Mesh{3}NTs{4}Dt{5}'.format(
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


def optcon_nse(N=24, Nts=10):

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
    # and setting current values
    newtk, t = 0, None

    # compute the steady state stokes solution
    rhsd_vfstbc = dict(fv=rhsd_stbc['fv'] +
                       rhsd_vf['fv'][INVINDS, ],
                       fp=rhsd_stbc['fp'] + rhsd_vf['fp'])

    vp_stokes = lau.stokes_steadystate(matdict=stokesmatsc,
                                       rhsdict=rhsd_vfstbc)

    # save the data
    curdatname = get_datastr(nwtn=newtk, time=t,
                             meshp=N, timps=tip)
    dou.save_npa(vp_stokes[:NV, ], fstring=ddir + 'vel' + curdatname)

#
# Compute the time-dependent flow
#

    # Stokes solution as initial value
    inivalvec = vp_stokes[:NV, ]

    norm_nwtnupd, newtk = 1, 1
    while newtk <= tip['nnewtsteps']:
        # check for previously computed velocities
        try:
            cdatstr = get_datastr(nwtn=newtk, time=tip['tE'],
                                  meshp=N, timps=tip)

            norm_nwtnupd = dou.load_npa(ddir + 'norm_nwtnupd' + cdatstr)
            prev_v = dou.load_npa(ddir + 'vel' + cdatstr)

            tip['norm_nwtnupd_list'].append(norm_nwtnupd)
            print 'loaded files of Newton iteration {0}'.format(newtk)
            print 'norm of current Newton update: {0}'.format(norm_nwtnupd[0])

            newtk += 1

        except IOError:
            break

    while (newtk <= tip['nnewtsteps'] and
           norm_nwtnupd > tip['vel_nwtn_tol']):

        set_vpfiles(tip, fstring=('results/' +
                                  'NewtonIt{0}').format(newtk))
        dou.output_paraview(tip, femp, vp=vp_stokes, t=0)

        norm_nwtnupd = 0
        v_old = inivalvec  # start vector in every Newtonit
        print 'Computing Newton Iteration {0} -- ({1} timesteps)'.\
            format(newtk, Nts)

        for t in np.linspace(tip['t0'] + DT, tip['tE'], Nts):
            cdatstr = get_datastr(nwtn=newtk, time=t,
                                  meshp=N, timps=tip)

            # t for implicit scheme
            pdatstr = get_datastr(nwtn=newtk - 1, time=t,
                                  meshp=N, timps=tip)

            # try - except for linearizations about stationary sols
            # for which t=None
            try:
                prev_v = dou.load_npa(ddir + 'vel' + pdatstr)
            except IOError:
                pdatstr = get_datastr(nwtn=newtk - 1, time=None,
                                      meshp=N, timps=tip)
                prev_v = dou.load_npa(ddir + 'vel' + pdatstr)

            # get and condense the linearized convection
            # rhsv_con += (u_0*D_x)u_0 from the Newton scheme
            N1, N2, rhs_con = dtn.get_convmats(u0_vec=prev_v,
                                               V=femp['V'],
                                               invinds=femp['invinds'],
                                               diribcs=femp['diribcs'])
            Nc, rhsv_conbc = dtn.condense_velmatsbybcs(N1 + N2,
                                                       femp['diribcs'])

            rhsd_cur = dict(fv=stokesmatsc['M'] * v_old +
                            DT * (rhs_con[INVINDS, :] +
                                  rhsv_conbc + rhsd_vfstbc['fv']),
                            fp=rhsd_vfstbc['fp'])

            matd_cur = dict(A=stokesmatsc['M'] +
                            DT * (stokesmatsc['A'] + Nc),
                            JT=stokesmatsc['JT'],
                            J=stokesmatsc['J'])

            vp = lau.stokes_steadystate(matdict=matd_cur,
                                        rhsdict=rhsd_cur)

            v_old = vp[:NV, ]

            dou.save_npa(v_old, fstring=ddir + 'vel' + cdatstr)

            dou.output_paraview(tip, femp, vp=vp, t=t),

            # integrate the Newton error
            norm_nwtnupd += DT * np.dot((v_old - prev_v).T,
                                        stokesmatsc['M'] *
                                        (v_old - prev_v))

        newtk += 1
        dou.save_npa(norm_nwtnupd, ddir + 'norm_nwtnupd' + cdatstr)
        tip['norm_nwtnupd_list'].append(norm_nwtnupd[0])

        print 'norm of current Newton update: {}'.format(norm_nwtnupd)

#
# Prepare for control
#

    # casting some parameters
    NY, NU = contp['NY'], contp['NU']

    contsetupstr = 'NV{0}NU{1}NY{2}'.format(NV, NU, NY)

    # get the control and observation operators
    try:
        b_mat = dou.load_spa(ddir + 'b_mat' + contsetupstr)
        u_masmat = dou.load_spa(ddir + 'u_masmat' + contsetupstr)
        print 'loaded `b_mat`'
    except IOError:
        print 'computing `b_mat`...'
        b_mat, u_masmat = cou.get_inp_opa(cdcoo=contp['cdcoo'],
                                          V=femp['V'], NU=contp['NU'])
        dou.save_spa(b_mat, ddir + 'b_mat' + contsetupstr)
        dou.save_spa(u_masmat, ddir + 'u_masmat' + contsetupstr)
    try:
        mc_mat = dou.load_spa(ddir + 'mc_mat' + contsetupstr)
        y_masmat = dou.load_spa(ddir + 'y_masmat' + contsetupstr)
        print 'loaded `c_mat`'
    except IOError:
        print 'computing `c_mat`...'
        MyC, y_masmat = cou.get_mout_opa(odcoo=contp['odcoo'],
                                         V=femp['V'], NY=contp['NY'])

        dou.save_spa(mc_mat, ddir + 'mc_mat' + contsetupstr)
        dou.save_spa(y_masmat, ddir + 'y_masmat' + contsetupstr)

    # restrict the operators to the inner nodes
    mc_mat = mc_mat[:, invinds][:, :]
    b_mat = b_mat[invinds, :][:, :]

    mct_mat_reg = cou.get_regularized_c(Ct=mc_mat.T, J=stokesmatsc['J'],
                                        Mt=stokesmatsc['MT'])

    # set the weighing matrices
    if contp['R'] is None:
        contp['R'] = contp['alphau'] * u_masmat
    # TODO: by now we tacitly assume that V, W = MyC.T My^-1 MyC
    # if contp['V'] is None:
    #     contp['V'] = My
    # if contp['W'] is None:
    #     contp['W'] = My

#
# solve the differential-alg. Riccati eqn for the feedback gain X
# via computing factors Z, such that X = -Z*Z.T
#
# at the same time we solve for the affine-linear correction w
#

    # cast some values from the dics
    ystar = contp['ystar']

    # tilde B = BR^{-1/2}
    tb_mat = lau.apply_invsqrt_fromleft(contp['R'], b_mat,
                                        output='sparse')

    trct_mat = lau.apply_invsqrt_fromleft(y_masmat, mct_mat_reg,
                                          output='dense')

    # set/compute the terminal values aka starting point
    Zc = lau.apply_massinv(stokesmatsc['M'], trct_mat)
    wc = -lau.apply_massinv(stokesmatsc['MT'], np.dot(mct_mat_reg, ystar))

    cdatstr = get_datastr(nwtn=newtk, time=tip['tE'], meshp=N, timps=tip)

    dou.save_npa(Zc, fstring=ddir + 'Z' + cdatstr)
    dou.save_npa(wc, fstring=ddir + 'w' + cdatstr)

    for t in np.linspace(tip['tE'] - DT, tip['t0'], Nts):
        # get the previous time convection matrices
        pdatstr = get_datastr(nwtn=newtk, time=t,
                              meshp=N, timps=tip)
        # try - except for linearizations about stationary sols
        # for which t=None
        try:
            prev_v = dou.load_npa(ddir + 'vel' + pdatstr)
        except IOError:
            pdatstr = get_datastr(nwtn=newtk, time=None,
                                  meshp=N, timps=tip)
            prev_v = dou.load_npa(ddir + 'vel' + pdatstr)

        # get and condense the linearized convection
        # rhsv_con += (u_0*D_x)u_0 from the Newton scheme
        N1, N2, rhs_con = dtn.get_convmats(u0_vec=prev_v, V=femp['V'],
                                           invinds=femp['invinds'],
                                           diribcs=femp['diribcs'])
        Nc, rhsv_conbc = dtn.condense_velmatsbybcs(N1 + N2,
                                                   femp['diribcs'])

        lyapAT = 0.5 * stokesmatsc['MT'] + DT * (stokesmatsc['AT'] + Nc.T)

        # rhs for nwtn adi
        w_mat = np.vstack([stokesmatsc['MT'] * Zc, np.sqrt(DT) * trct_mat])

        Zp = pru.proj_alg_ric_newtonadi(mmat=stokesmatsc['MT'],
                                        fmat=lyapAT,
                                        transposed=True,
                                        jmat=stokesmatsc['J'],
                                        bmat=np.sqrt(DT)*tb_mat,
                                        wmat=w_mat,
                                        z0=Zc,
                                        nwtn_adi_dict=tip['nwtn_adi_dict']
                                        )['zfac']

    dou.save_npa(Zp, fstring=ddir + 'Z' + cdatstr)
    Zc = Zp


if __name__ == '__main__':
    optcon_nse()
