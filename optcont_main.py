from dolfin import *
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os, glob

import dolfin_to_nparrays as dtn
import linsolv_utils
import data_output_utils as dou
import cont_obs_utils as cou
import proj_ric_utils as pru

parameters.linear_algebra_backend = 'uBLAS'

def time_int_params(Nts):
    t0 = 0.0
    tE = 1.0
    dt = (tE - t0)/Nts
    tip = dict(t0 = t0,
            tE = tE,
            dt = dt, 
            Nts = Nts,
            # vpfiles = def_vpfiles(), 
            Residuals = NseResiduals(), 
            ParaviewOutput = True, 
            nu = 1e-3,
            nnewtsteps = 1,
            norm_nwtnupd = [],
            # parameters for the Newton-ADI
            nnwtadisteps = 2,
            nadisteps = 10
            )

    return tip

def optcon_nse(N = 20, Nts = 4, compvels=False):

    tip = time_int_params(Nts)
    femp = drivcav_fems(N)
    contp = cont_params()

    ### Output
    ddir = 'data/'
    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdirectory for storing the data')
    os.chdir('..')

    #if tip['ParaviewOutput']:
    #    os.chdir('results/')
    #    for fname in glob.glob(TsP.method + '*'):
    #        os.remove( fname )
    #    os.chdir('..')

###
## start with the Stokes problem for initialization
###

    stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'],
                                         tip['nu'])
    rhsd_vf = dtn.setget_rhs(femp['V'], femp['Q'], 
                            femp['fv'], femp['fp'], t=0)

    # remove the freedom in the pressure 
    stokesmats['J'] = stokesmats['J'][:-1,:][:,:]
    stokesmats['JT'] = stokesmats['JT'][:,:-1][:,:]
    rhsd_vf['fp'] = rhsd_vf['fp'][:-1,:]

    # reduce the matrices by resolving the BCs
    (stokesmatsc, 
            rhsd_stbc, 
            invinds, 
            bcinds, 
            bcvals) = dtn.condense_sysmatsbybcs(stokesmats, femp['diribcs'])
    # we will need transposes, and explicit is better than implicit
    # here, the coefficient matrices are symmetric
    stokesmatsc.update(dict(MT=stokesmatsc['M'],
                            AT=stokesmatsc['A']))

    # add the info on boundary and inner nodes 
    bcdata = {'bcinds':bcinds,
            'bcvals':bcvals,
            'invinds':invinds}
    femp.update(bcdata)

    # casting some parameters 
    NV, DT, INVINDS = len(femp['invinds']), tip['dt'], femp['invinds']
    # and setting current values
    newtk, t = 0, None

    # compute the steady state stokes solution
    rhsd_vfstbc = dict(fv=rhsd_stbc['fv']+
                            rhsd_vf['fv'][INVINDS,],
                       fp=rhsd_stbc['fp']+rhsd_vf['fp'])
    vp = linsolv_utils.stokes_steadystate(matdict=stokesmatsc,
                                        rhsdict=rhsd_vfstbc)

    # save the data
    curdatname = get_datastr(nwtn=newtk, time=t, 
                              meshp=N, timps=tip)
    dou.save_curv(vp[:NV,], fstring=ddir+'vel'+curdatname)

    dou.output_paraview(femp, vp=vp, 
                    fstring='results/'+
                            'NewtonIt{0}'.format(newtk))

    # Stokes solution as initial value
    inivalvec = vp[:NV,]

###
## Compute the time-dependent flow
###

    if compvels:
        for newtk in range(1, tip['nnewtsteps']+1):
            v_old = inivalvec
            norm_nwtnupd = 0
            for t in np.arange(tip['t0'], tip['tE'], DT):
                cdatstr = get_datastr(nwtn=newtk, time=t+DT, 
                                      meshp=N, timps=tip)

                # t+DT for implicit scheme
                pdatstr = get_datastr(nwtn=newtk-1, time=t+DT, 
                                         meshp=N, timps=tip)

                # try - except for linearizations about stationary sols
                # for which t=None
                try:
                    prev_v = dou.load_curv(ddir+'vel'+pdatstr)
                except IOError:
                    pdatstr = get_datastr(nwtn=newtk-1, time=None, 
                                             meshp=N, timps=tip)
                    prev_v = dou.load_curv(ddir+'vel'+pdatstr)

                # get and condense the linearized convection
                # rhsv_con += (u_0*D_x)u_0 from the Newton scheme
                N1, N2, rhs_con = dtn.get_convmats(u0_vec=prev_v,
                                            V=femp['V'],
                                            invinds=femp['invinds'],
                                            diribcs = femp['diribcs'])
                Nc, rhsv_conbc = dtn.condense_velmatsbybcs(N1+N2,
                                            femp['diribcs'])

                rhsd_cur = dict(fv=stokesmatsc['M']*v_old+
                                    DT*(rhs_con[INVINDS,:]+
                                    rhsv_conbc+rhsd_vfstbc['fv']),
                                fp=rhsd_vfstbc['fp'])

                matd_cur = dict(A=stokesmatsc['M']+
                                            DT*(stokesmatsc['A']+Nc),
                                JT=stokesmatsc['JT'],
                                J=stokesmatsc['J'])

                vp = linsolv_utils.stokes_steadystate(matdict=matd_cur,
                                                      rhsdict=rhsd_cur)

                v_old = vp[:NV,]

                dou.save_curv(v_old, fstring=ddir+'vel'+cdatstr)

                norm_nwtnupd += DT*np.dot((v_old-prev_v).T, 
                                    stokesmatsc['M']*(v_old-prev_v))

            tip['norm_nwtnupd'].append(norm_nwtnupd)

        print tip['norm_nwtnupd']

###
## Prepare for control 
###
    # casting some parameters
    NY = contp['NY']

    # define the control and observation operators
    cdom = cou.ContDomain(contp['cdcoo'])
    Ba, Mu = cou.get_inp_opa(cdom=cdom, V=femp['V'],
                             NU=contp['NU']) 
    Ba = Ba[invinds,:][:,:]

    odom = cou.ContDomain(contp['odcoo'])
    MyC, My = cou.get_mout_opa(odom=odom, V=femp['V'],
                               NY=contp['NY'])
    MyC = MyC[:,invinds][:,:]
    C = cou.get_regularized_c(Ct=MyC.T, J=stokesmatsc['J'], 
                            Mt=stokesmatsc['MT'])

    # we never need vstar, do we?
    # vstar = cou.get_vstar(C, contp['ystar'], contp['odcoo'], NY)

    # set the weighing matrices
    if contp['R'] is None:
        contp['R'] = contp['alphau']*Mu
    # TODO: by now we tacitly assume that V, W = MyC.T My^-1 MyC
    # if contp['V'] is None:
    #     contp['V'] = My
    # if contp['W'] is None:
    #     contp['W'] = My

###
## solve the differential-alg. Riccati eqn for the feedback gain X
## via computing factors Z, such that X = -Z*Z.T
## 
## at the same time we solve for the affine-linear correction w
###
    
    # cast some values from the dics
    ystar = contp['ystar']

    # tB = BR^{-1/2}
    tB = linsolv_utils.apply_invsqrt_fromleft(contp['R'], Ba,
                                              output='sparse')
    tCT = linsolv_utils.apply_invsqrt_fromleft(My, MyC.T, 
                                                output='dense')

    t = tip['tE']

    # set/compute the terminal values
    Zc = linsolv_utils.apply_massinv(stokesmatsc['M'], tCT)
    wc = -linsolv_utils.apply_massinv(stokesmatsc['MT'], MyC.T*ystar)

    cdatstr = get_datastr(nwtn=newtk, time=DT, meshp=N, timps=tip)

    dou.save_curv(Zc, fstring=ddir+'Z'+cdatstr) 

    for t in np.linspace(tip['tE']-DT, tip['t0'],
                            np.round((tip['tE']-tip['t0'])/DT)):
        # get the previous time convection matrices 
        pdatstr = get_datastr(nwtn=newtk, time=t, 
                              meshp=N, timps=tip)
        # try - except for linearizations about stationary sols
        # for which t=None
        try:
            prev_v = dou.load_curv(ddir+'vel'+pdatstr)
        except IOError:
            pdatstr = get_datastr(nwtn=newtk, time=None, 
                                     meshp=N, timps=tip)
            prev_v = dou.load_curv(ddir+'vel'+pdatstr)
        # get and condense the linearized convection
        # rhsv_con += (u_0*D_x)u_0 from the Newton scheme
        N1, N2, rhs_con = dtn.get_convmats(u0_vec=prev_v, V=femp['V'],
                                        invinds=femp['invinds'],
                                        diribcs=femp['diribcs'])
        Nc, rhsv_conbc = dtn.condense_velmatsbybcs(N1+N2,
                                                    femp['diribcs'])

        lyapAT = 0.5*stokesmatsc['MT'] + DT*(stokesmatsc['AT'] + Nc.T)

        # starting value for Newton-ADI iteration
        Zpn = np.copy(Zc)


    dou.save_curv(Zpn, fstring=ddir+'Z'+cdatstr) 
    Zc = Zpn


def drivcav_fems(N, NU=None, NY=None):
    """dictionary for the fem items of the (unit) driven cavity

    """
    mesh = UnitSquareMesh(N, N)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    # pressure node that is set to zero

    # Boundaries
    def top(x, on_boundary): 
        return x[1] > 1.0 - DOLFIN_EPS 
    def leftbotright(x, on_boundary): 
        return ( x[0] > 1.0 - DOLFIN_EPS 
            or x[1] < DOLFIN_EPS 
            or x[0] < DOLFIN_EPS)

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0))
    bc0 = DirichletBC(V, noslip, leftbotright)
    # Boundary condition for velocity at the lid
    lid = Constant(("1", "0.0"))
    bc1 = DirichletBC(V, lid, top)
    # Collect boundary conditions
    diribcs = [bc0, bc1]
    # rhs of momentum eqn
    fv = Constant((0.0, 0.0))
    # rhs of the continuity eqn
    fp = Constant(0.0)

    dfems = dict(mesh = mesh,
            V = V,
            Q = Q,
            diribcs = diribcs,
            fv = fv,
            fp = fp)

    return dfems


def cont_params():
    """define the parameters of the control problem 

    as there are
    - dimensions of in and output space
    - extensions of the subdomains of control and observation
    - weighting matrices (if None, then massmatrix)
    - desired output
    """

    ystar1 = Expression('1')
    ystar2 = Expression('1')

    ystar = [ystar1, ystar2]

    NU, NY = 10, 7
    odcoo = dict(xmin=0.45,
                 xmax=0.55,
                 ymin=0.5,
                 ymax=0.7)

    cdcoo = dict(xmin=0.4,
                 xmax=0.6,
                 ymin=0.2,
                 ymax=0.3)

    return dict(NU=NU,
                NY=NY,
                cdcoo=cdcoo,
                odcoo=odcoo,
                R=None, 
                # regularization parameter
                alphau=1e-4,
                ystar=ystar,
                V=None,
                W=None)

class NseResiduals(object):
    def __init__(self):
        self.ContiRes = []
        self.VelEr = []
        self.PEr = []


#def def_vpfiles(name=None):
#    if name is not None:
#        vpf = {'vfile':File("results/%s_velocity.pvd" % name), 
#                'pfile':File("results/%s_pressure.pvd" % name)}
#    else:
#        vpf = {'vfile':File("results/velocity.pvd"), 
#                'pfile':File("results/pressure.pvd")}
#
#    return vpf

def get_datastr(nwtn=None, time=None, 
                            meshp=None, timps=None):
    return 'NIt{0}Time{1}Mesh{2}NTs{3}Dt{4}'.format(
            nwtn, time, meshp,
            timps['Nts'],timps['dt']
            )

if __name__ == '__main__':
    optcon_nse()
