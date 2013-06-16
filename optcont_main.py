from dolfin import *
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os, glob

import dolfin_to_nparrays as dtn
import solvers_drivcav 
import data_output_utils as dou

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
            nnewtsteps = 6,
            norm_nwtnupd = []
            )

    return tip

def optcon_nse(N = 24, Nts = 10):

    tip = time_int_params(Nts)
    femp = drivcav_fems(N)

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
    dou.save_curv(vp[:NV,], fstring=ddir+curdatname)

    dou.output_paraview(femp, vp=vp, 
                    fstring='results/'+'NewtonIt{0}'.format(newtk))

    # Stokes solution as initial value
    inivalvec = vp[:NV,]

    for newtk in range(1, tip['nnewtsteps']+1):
        v_old = inivalvec
        norm_newtondif = 0
        for t in np.arange(tip['t0'], tip['tE'], DT):
            cdatstr = get_datastr(nwtn=newtk, time=t+DT, 
                                  meshp=N, timps=tip)

            # t+DT for implicit scheme
            pdatstr = get_datastr(nwtn=newtk-1, time=t+DT, 
                                     meshp=N, timps=tip)

            # try - except for linearizations about stationary sols
            # for which t=None
            try:
                prev_v = dou.load_curv(ddir+pdatstr)
            except IOError:
                pdatstr = get_datastr(nwtn=newtk-1, time=None, 
                                         meshp=N, timps=tip)
                prev_v = dou.load_curv(ddir+pdatstr)

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
                            BT=stokesmatsc['BT'],
                            B=stokesmatsc['B'])

            vp = solvers_drivcav.stokes_steadystate(matdict=matd_cur,
                                                    rhsdict=rhsd_cur)

            v_old = vp[:NV,]

            dou.save_curv(v_old, fstring=ddir+cdatstr)

            norm_newtondif += DT*np.dot((v_old-prev_v).T, 
                                        stokesmatsc['M']*(v_old-prev_v))

        tip['norm_newtondif'].append(norm_newtondif)


def drivcav_fems(N):
    """dictionary for the fem items of the (unit) driven cavity

    """
    mesh = UnitSquare(N, N)
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
