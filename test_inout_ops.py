from dolfin import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import cont_obs_utils as cou
parameters.linear_algebra_backend = "uBLAS"

N = 40
mesh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(mesh, "CG", 2)

NU, NY = 12, 14

cdcoo = dict(xmin=0.4,
             xmax=0.6,
             ymin=0.2,
             ymax=0.3)

odcoo = dict(xmin=0.45,
             xmax=0.55,
             ymin=0.6,
             ymax=0.8)

cdom = cou.ContDomain(cdcoo)
odom = cou.ContDomain(odcoo)

# get mass mat
v, w = TrialFunction(V), TestFunction(V)
Mv = assemble(inner(w,v)*dx)
rows, cols, values = Mv.data()
Mv = sps.csr_matrix((values, cols, rows))

# # check the B
# B = cou.get_inp_opa(cdom=cdom, V=V, NU=NU) 
# 
# # get mass mat
# v, w = TrialFunction(V), TestFunction(V)
# Mv = assemble(inner(w,v)*dx)
# rows, cols, values = Mv.data()
# Mv = sps.csr_matrix((values, cols, rows))
# Bu = spsla.spsolve(Mv,B*np.vstack([0*np.ones((NU,1)),
#                                    1*np.ones((NU,1))]))
# 
# v = Function(V)
# v.vector().set_local(Bu)
# plot(v)

# check the C

MyC, My = cou.get_mout_opa(odom=odom, V=V, NY=NY)

exv = Expression(('1', '1'))
testv = interpolate(exv, V)

plot(testv)

MyCv = MyC*testv.vector().array()
testy = spsla.spsolve(My,MyCv)

# print np.linalg.norm(testy)

ymesh = IntervalMesh(NY-1, odcoo['ymin'], odcoo['ymax'])
Y = FunctionSpace(ymesh, 'CG', 1)
yx = Function(Y)
yy = Function(Y)

yx.vector().set_local(testy[:NY])
plot(yx)
yy.vector().set_local(testy[NY:])
plot(yy)

interactive(True)
