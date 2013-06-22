from dolfin import *
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os, glob

import dolfin_to_nparrays as dtn
import linsolv_utils
import data_output_utils as dou
import cont_obs_utils as cou

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
            nnwtadisteps = 3
            )

    return tip

def optcon_nse(N = 20, Nts = 10):

    tip = time_int_params(Nts)
    femp = drivcav_fems(N)
    contp = inout_params()

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

    ## start with the Stokes problem for initialization
    stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], tip['nu'])
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
    # add the info on boundary and inner nodes 
    bcdata = {'bcinds':bcinds,
            'bcvals':bcvals,
            'invinds':invinds}
    femp.update(bcdata)

    # define the control and observation operators
    cdom = cou.ContDomain(contp['cdcoo'])
    Ba, Mu = cou.get_inp_opa(cdom=cdom, V=femp['V'], NU=contp['NU']) 
    Ba = Ba[invinds,:][:,:]

    odom = cou.ContDomain(contp['odcoo'])
    MyC, My = cou.get_mout_opa(odom=odom, V=femp['V'], NY=contp['NY'])
    MyC = MyC[:,invinds][:,:]
    C = cou.get_regularzd_c(MyC, My, J=stokesmatsc['J'], M=stokesmatsc['M'])
    # raise Warning('STOP: in the name of love') 

    # casting some parameters 
    NV, DT, INVINDS = len(femp['invinds']), tip['dt'], femp['invinds']
    # and current values
    newtk, t = 0, None

    # compute the steady state stokes solution
    rhsd_vfstbc = dict(fv=rhsd_stbc['fv']+rhsd_vf['fv'][INVINDS,],
                        fp=rhsd_stbc['fp']+rhsd_vf['fp'])
    vp = solvers_drivcav.stokes_steadystate(matdict=stokesmatsc,
                                        rhsdict=rhsd_vfstbc)

    # save the data
    curdatname = get_datastr(nwtn=newtk, time=t, meshp=N, timps=tip)
    dou.save_curv(vp[:NV,], fstring=ddir+'vel'+curdatname)

    dou.output_paraview(femp, vp=vp, 
                    fstring='results/'+'NewtonIt{0}'.format(newtk))

    # Stokes solution as initial value
    inivalvec = vp[:NV,]

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
            N1, N2, rhs_con = dtn.get_convmats(u0_vec=prev_v, V=femp['V'],
                                            invinds=femp['invinds'],
                                            diribcs = femp['diribcs'])
            Nc, rhsv_conbc = dtn.condense_velmatsbybcs(N1+N2,
                                                        femp['diribcs'])

            rhsd_cur = dict(fv=stokesmatsc['M']*v_old+
                                DT*(rhs_con[INVINDS,:]+
                                rhsv_conbc+rhsd_vfstbc['fv']),
                            fp=rhsd_vfstbc['fp'])

            matd_cur = dict(A=stokesmatsc['M']+DT*(stokesmatsc['A']+Nc),
                            JT=stokesmatsc['JT'],
                            J=stokesmatsc['J'])

            vp = solvers_drivcav.stokes_steadystate(matdict=matd_cur,
                                                    rhsdict=rhsd_cur)

            v_old = vp[:NV,]

            dou.save_curv(v_old, fstring=ddir+'vel'+cdatstr)

            norm_nwtnupd += DT*np.dot((v_old-prev_v).T, 
                                        stokesmatsc['M']*(v_old-prev_v))

        tip['norm_nwtnupd'].append(norm_nwtnupd)

    print tip['norm_nwtnupd']

## solve the differential-alg. Riccati eqn for the feedback gain X
#  via computing factors Z, such that X = -Z*Z.T

    # tB = BR^{-1/2}
    tB = linsolv_utils.apply_massinvsqrtfromleft(R, B)
    tCT = linsolv_utils.apply_massinvsqrtfromleft(My, MyC.T)

    t = tE
    Zp = linsolv_utils.apply_massinv(stokesmatsc['M'], tCT)

    cdatstr = get_datastr(nwtn=newtk, time=DT, 
                          meshp=N, timps=tip)

    dou.save_curv(Z, fstring=ddir+'Z'+cdatstr) 

    for t in np.linspace(tE-DT, t0, np.round((tE-t0)/DT)):
        Zcn = np.copy(Zp) # starting value for Newton-ADI iteration
        for nnwtadi in range(tip['nnwtadisteps']):
            cmtxtb = M.T*np.dot(Zcn, Zcn.T*tB)
            crhsadi = np.hstack([stokesmatsc['M']*Zp,
                      np.hstack([np.sqrt(DT)*cmtxtb, tCT])])

            # get the current convection matrices 
            cdatstr = get_datastr(nwtn=newtk, time=t, 
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
            N1, N2, rhs_con = dtn.get_convmats(u0_vec=prev_v, V=femp['V'],
                                            invinds=femp['invinds'],
                                            diribcs = femp['diribcs'])
            Nc, rhsv_conbc = dtn.condense_velmatsbybcs(N1+N2,
                                                        femp['diribcs'])


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


def inout_params():
    """define the extensions of the subdomains

    of control and observation"""

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
                odcoo=odcoo)


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
