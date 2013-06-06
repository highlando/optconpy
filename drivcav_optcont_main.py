from dolfin import *
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os, glob

import dolfin_to_nparrays as dtn

class TimeIntParams(object):
    def __init__(self, Nts):
        self.t0 = 0.0
        self.tE = 1.0
        self.dt = (self.tE - self.t0)/Nts
        self.UpFiles = UpFiles()
        self.Residuals = NseResiduals()
        self.ParaviewOutput = False
        self.SaveTStps = False

def optcon_nse(N = 12, Nts = 10):
    tip = TimeIntParams(Nts)
    dcf = drivcav_femform(N)

    V, Q, f, diribcs = dcf.V, dcf.Q, dcf.f, dcf.diribcs


#ufile_pvd = File("velocity.pvd")
#ufile_pvd << u
#pfile_pvd = File("pressure.pvd")
#pfile_pvd << p
#
## Plot solution
#plot(u)
#plot(p)
#interactive()

class drivcav_femform(object):
    """container for the fem items of the (unit) driven cavity

    """
    def __init__(self, N):
        self.mesh = UnitSquareMesh(N, N)
        self.V = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Q = FunctionSpace(self.mesh, "CG", 1)

        # Boundaries
        def top(x, on_boundary): 
            return x[1] > 1.0 - DOLFIN_EPS 
        def leftbotright(x, on_boundary): 
            return ( x[0] > 1.0 - DOLFIN_EPS 
                or x[1] < DOLFIN_EPS 
                or x[0] < DOLFIN_EPS)

        # No-slip boundary condition for velocity
        noslip = Constant((0.0, 0.0))
        bc0 = DirichletBC(self.V, noslip, leftbotright)
        # Boundary condition for velocity at the lid
        lid = Constant(("1", "0.0"))
        bc1 = DirichletBC(self.V, lid, top)
        # Collect boundary conditions
        self.diribcs = [bc0, bc1]
        # rhs of the continuity eqn
        self.f = Constant((0.0, 0.0))


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
