from dolfin import *
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os, glob

import dolfin_to_nparrays as dtn
import solvers_drivcav 

parameters.linear_algebra_backend = 'uBLAS'

def time_int_params(Nts):
    t0 = 0.0
    tE = 1.0
    dt = (tE - t0)/Nts

    tip = dict(t0 = t0,
            tE = tE,
            dt = dt, 
            UpFiles = UpFiles(), 
            Residuals = NseResiduals(), 
            ParaviewOutput = False, 
            SaveTStps = False, 
            nu = 1e-4)

    return tip

def optcon_nse(N = 32, Nts = 10):

    tip = time_int_params(Nts)
    femp = drivcav_fems(N)

    stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], tip['nu'])

    rhsvecs = dtn.setget_rhs(femp['V'], femp['Q'], femp['fv'], femp['fp'], t=0)

    stokesmatsc, rhsvecbc, bcdata = dtn.condense_sysmatsbybcs(stokesmats,
            rhsvecs, femp['diribcs'])

    # add the info on boundary and inner nodes 
    femp.update(bcdata)

    solvers_drivcav.stokes_steadystate(stokesmatsc, rhsvecbc, femp, tip)



#ufile_pvd = File("velocity.pvd")
#ufile_pvd << u
#pfile_pvd = File("pressure.pvd")
#pfile_pvd << p
#

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

class UpFiles(object):
        def __init__(self, name=None):
            if name is not None:
                self.u_file = File("results/%s_velocity.pvd" % name)
                self.p_file = File("results/%s_pressure.pvd" % name)
            else:
                self.u_file = File("results/velocity.pvd")
                self.p_file = File("results/pressure.pvd")

if __name__ == '__main__':
    optcon_nse()
