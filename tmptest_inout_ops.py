# this is rather optical checking
import dolfin
import numpy as np
import scipy.sparse.linalg as spsla

import dolfin_to_nparrays as dtn
import cont_obs_utils as cou

from optcont_main import drivcav_fems
dolfin.parameters.linear_algebra_backend = "uBLAS"

N = 20
mesh = dolfin.UnitSquareMesh(N, N)
V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
Q = dolfin.FunctionSpace(mesh, "CG", 1)

# test velocity
# case 1 -- not div free
# exv = dolfin.Expression(('x[0]', '-x[1]'))
# case 2 -- div free
# exv = dolfin.Expression(('1', '1'))

import sympy as smp
x, y = smp.symbols('x[0], x[1]')
u_x = x*x*(1-x)*(1-x)*2*y*(1-y)*(2*y-1)
u_y = y*y*(1-y)*(1-y)*2*x*(1-x)*(1-2*x)
from sympy.printing import ccode
exv = dolfin.Expression((ccode(u_x), ccode(u_y)))


testv = dolfin.interpolate(exv, V)

NU, NY = 12, 7

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

# get the system matrices
femp = drivcav_fems(N)
stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], nu=1)
# remove the freedom in the pressure
stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]

# bcinds = []
bc = dolfin.DirichletBC(V, exv, 'on_boundary')
# bcdict = bc.get_boundary_values()
# auxu[bcdict.keys(),0] = bcdict.values()
# bcinds.extend(bcdict.keys())
# # indices of the innernodes
# invinds = np.setdiff1d(range(nv),bcinds).astype(np.int32)

# reduce the matrices by resolving the BCs
(stokesmatsc,
 rhsd_stbc,
 invinds,
 bcinds,
 bcvals) = dtn.condense_sysmatsbybcs(stokesmats, [bc])

# check the B
B, Mu = cou.get_inp_opa(cdom=cdom, V=V, NU=NU)
# get the rhs expression of Bu
Bu = spsla.spsolve(stokesmats['M'], B*np.vstack([0*np.ones((NU, 1)),
                                                 1*np.ones((NU, 1))]))

bu = dolfin.Function(V)
bu.vector().set_local(Bu)
dolfin.plot(bu, title='plot of Bu')

# check the C
MyC, My = cou.get_mout_opa(odom=odom, V=V, NY=NY)
MyC = MyC[:, invinds][:, :]

# testsignal from the test velocity
MyCv = MyC * testv.vector().array()[invinds]
testy = spsla.spsolve(My, MyCv)

# target signal
# ystar1 = dolfin.Expression('x[0]')
# ystar2 = dolfin.Expression('x[0]')
# ystar = [ystar1, ystar2]

# gives a target velocity
# vstar = cou.get_vstar(MyC, ystar, odcoo, NY)

# print np.linalg.norm(vstar)
# print np.linalg.norm(MyC * vstar)

# signal space
ymesh = dolfin.IntervalMesh(NY - 1, odcoo['ymin'], odcoo['ymax'])
Y = dolfin.FunctionSpace(ymesh, 'CG', 1)

y1 = dolfin.Function(Y)
y2 = dolfin.Function(Y)
y3 = dolfin.Function(Y)
y4 = dolfin.Function(Y)
y5 = dolfin.Function(Y)
y6 = dolfin.Function(Y)

y1.vector().set_local(testy[:NY])
y1.rename("x-comp of C*v", "signal")
dolfin.plot(y1)

# y2.vector().set_local(testy[NY:])
# y2.rename("y-comp of testsignal", "signal")
# dolfin.plot(y2)

# check the regularization of C
rC = cou.get_regularized_c(MyC.T, J=stokesmatsc['J'], Mt=stokesmatsc['M']).T

testvi = testv.vector().array()[invinds]
testvi0 = cou.app_difffreeproj(
    M=stokesmatsc['M'],
    J=stokesmatsc['J'],
    v=testvi)

print np.linalg.norm(stokesmatsc['J'] * testvi)
print np.linalg.norm(stokesmatsc['J'] * testvi0)

testyv0 = spsla.spsolve(My, MyC * testvi0)
testyg = spsla.spsolve(My, MyC * (testvi.flatten() - testvi0))
testry = spsla.spsolve(My, np.dot(rC, testvi))

# testystar = MyC * vstar

print np.linalg.norm(testyv0 - testry)

y3.vector().set_local(testry[NY:])
dolfin.plot(y3, title='x-comp of $(C*P_{df})v$')

y4.vector().set_local(testyv0[NY:])
dolfin.plot(y4, title="x-comp of $C*(P_{df}v)$")

y5.vector().set_local(testyg[NY:])
dolfin.plot(y5, title="x-comp of $C*(v - P_{df}v)$")

# y6.vector().set_local(testystar[NY:])
# dolfin.plot(y6, title="x-comp of $C*C^+y^*$")

# check if the projection is indeed a projection
# os.remove('data/regCNY14vdim3042.npy')
# Ci = cou.get_regularized_c(sps.csr_matrix(My*C).T, J=stokesmatsc['J'],
#                             Mt=stokesmatsc['M']).T
# print np.linalg.norm(np.dot(C,testvi) - np.dot(Ci,testvi))

dolfin.interactive(True)
