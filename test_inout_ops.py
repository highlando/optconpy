from dolfin import *
import numpy as np
import scipy.sparse as sps

import cont_obs_utils as cou

N = 48 
mesh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(mesh, "CG", 2)

NU = 2

cdcoo = dict(xmin=0.4,
            xmax=0.6,
            ymin=0.2,
            ymax=0.3)

cdom = cou.ContDomain(cdcoo)

B = cou.get_inp_opa(cdom=cdom, V=V, NU=NU) 


