from dolfin import *
import numpy as np
import scipy.sparse as sps
import cont_obs_utils as cou


N = 24
mesh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(mesh, "CG", 2)

NU = 8

cdcoo = dict(xmin=0.4,
            xmax=0.6,
            ymin=0.2,
            ymax=0.3)

cdom = ContDomain(cdcoo)

xv = np.arange(cdcoo['xmin'], cdcoo['xmax'], 
                (cdcoo['xmax']-cdcoo['xmin'])/20)

vv = np.zeros(2)
bvv = np.zeros(2)

u1 = L2abLinBas(3,NU)
Bu = Inp2Rhs2D(u1, cdom)

for x in xv:
    u1.evaluate(vv, np.array([x]))
    Bu.eval(bvv, np.array([x,0.25]))
    print x, vv, bvv

return u1, Bu, V

