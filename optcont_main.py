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
            nu = 1e-4,
            nnewtsteps = 3
            )

    return tip

def optcon_nse(N = 32, Nts = 10):

    tip = time_int_params(Nts)
    femp = drivcav_fems(N)

    ### Output
    try:
        os.chdir('data')
    except OSError:
        raise Warning('need "data" subdirectory for storing the data')
    os.chdir('..')

    #if tip['ParaviewOutput']:
    #    os.chdir('results/')
    #    for fname in glob.glob(TsP.method + '*'):
    #        os.remove( fname )
    #    os.chdir('..')

    ## start with the Stokes problem for initialization
    stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], tip['nu'])
    rhsvecs = dtn.setget_rhs(femp['V'], femp['Q'], 
                            femp['fv'], femp['fp'], t=0)
    # reduce the matrices by resolving the BCs
    stokesmatsc, rhsvecbc, innerinds, bcinds, bcvals = dtn.condense_sysmatsbybcs(
                    stokesmats, rhsvecs, femp['diribcs'])
    # add the info on boundary and inner nodes 
    bcdata = {'bcinds':bcinds,
            'bcvals':bcvals,
            'innerinds':innerinds}
    femp.update(bcdata)

    # some parameters 
    NV = len(femp['innerinds'])
    # and current values
    newtK, t = 0, None

    # compute the steady state stokes solution
    vp = solvers_drivcav.stokes_steadystate(matdict=stokesmatsc,
                                        rhsdict=rhsvecbc)

    # save the data
    curdatname = get_cur_datastring(newtonstp=newtk, time=t, 
                                    meshpara=N, timeinitparams=tip)
    dou.save_curv(vp[:,:NV], 
                    fstring='data/'+curdatname)
    dou.output_paraview(femp, vp=vp, 
                    fstring='results/'+'NewtonIt{0}'.format(newtK))

    # Stokes solution as initial value
    inivalvec = vp[:,:NV]

    for newtk in range(1, tip['nnewtsteps']+1):



def drivcav_fems(N):
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

def get_cur_datastring(newtonstp=None,
                        time=None,
                        meshpara=None,
                        timeinitparams=None):
    return 'NIt{0}Time{1}Mesh{2}NTs{3}Dt{4}'.format(
            newtonstp,time,meshpara,
            timeinitparams['Nts'],timeinitparams['dt']
            )

if __name__ == '__main__':
    optcon_nse()
