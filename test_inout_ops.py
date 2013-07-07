# this is rather optical checking 

from dolfin import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import dolfin_to_nparrays as dtn
import cont_obs_utils as cou
import os

from optcont_main import drivcav_fems
parameters.linear_algebra_backend = "uBLAS"

N = 20
mesh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

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

## get the system matrices
femp = drivcav_fems(N)
stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], 1)
# remove the freedom in the pressure 
stokesmats['J'] = stokesmats['J'][:-1,:][:,:]
stokesmats['JT'] = stokesmats['JT'][:,:-1][:,:]
# reduce the matrices by resolving the BCs
(stokesmatsc, 
        rhsd_stbc, 
        invinds, 
        bcinds, 
        bcvals) = dtn.condense_sysmatsbybcs(stokesmats, femp['diribcs'])

# ## check the B
# B, Mu = cou.get_inp_opa(cdom=cdom, V=V, NU=NU) 
# # get the rhs expression of Bu
# Bu = spsla.spsolve(stokesmats['M'], B*np.vstack([0*np.ones((NU,1)),
#                                                  1*np.ones((NU,1))]))
# 
# bu = Function(V)
# bu.vector().set_local(Bu)
# plot(bu)

## check the C
MyC, My = cou.get_mout_opa(odom=odom, V=V, NY=NY)
MyC = MyC[:,invinds][:,:]

exv = Expression(('x[1]', 'x[1]'))
testv = interpolate(exv, V)
# plot(testv)

Cplus = cou.get_rightinv(MyC)

print np.allclose(np.eye(MyC.shape[0]), MyC*Cplus)

MyCv = MyC*testv.vector().array()[invinds]
testy = spsla.spsolve(My, MyCv)
# print np.linalg.norm(testy)

ystar1 = Expression('x[0]')
ystar2 = Expression('x[0]')
ystar = [ystar1, ystar2]

vstar = cou.get_vstar(MyC, ystar, odcoo, NY)
print np.linalg.norm(vstar)
print np.linalg.norm(MyC*vstar)

ymesh = IntervalMesh(NY-1, odcoo['ymin'], odcoo['ymax'])
Y = FunctionSpace(ymesh, 'CG', 1)
y1 = Function(Y)
y2 = Function(Y)
y3 = Function(Y)
y4 = Function(Y)
y5 = Function(Y)
y6 = Function(Y)

# interactive(True)
y1.vector().set_local(testy[:NY])
plot(y1)
y2.vector().set_local(testy[NY:])
plot(y2)

## check the regularization of C
# MyC = MyC[:,invinds][:,:]
rC = cou.get_regularized_c(MyC.T, J=stokesmatsc['J'], Mt=stokesmatsc['M']).T

testvi = testv.vector().array()[invinds]
testvi0 = cou.app_difffreeproj(M=stokesmatsc['M'],J=stokesmatsc['J'],v=testvi)

print np.linalg.norm(stokesmatsc['J']*testvi)
print np.linalg.norm(stokesmatsc['J']*testvi0)

testyv0 = spsla.spsolve(My, MyC*testvi0)
testyg = spsla.spsolve(My, MyC*(testvi.flatten()-testvi0))
testry = spsla.spsolve(My, np.dot(rC, testvi))
testystar = MyC*vstar

print np.linalg.norm(testyv0 - testry)

y3.vector().set_local(testry[NY:])
plot(y3, title='rCv')
y4.vector().set_local(testyv0[NY:])
plot(y4, title='Cv0')
y5.vector().set_local(testyg[NY:])
plot(y5, title='Cvg')
y6.vector().set_local(testystar[NY:])
plot(y6, title='ystar')

## check if the projection is indeed a projection
# os.remove('data/regCNY14vdim3042.npy')
# Ci = cou.get_regularized_c(sps.csr_matrix(My*C).T, J=stokesmatsc['J'],
#                             Mt=stokesmatsc['M']).T
# print np.linalg.norm(np.dot(C,testvi) - np.dot(Ci,testvi))

