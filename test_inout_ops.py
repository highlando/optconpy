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
Q = FunctionSpace(mesh, "CG", 2)

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

# get mass mat
v, w = TrialFunction(V), TestFunction(V)
Mv = assemble(inner(w,v)*dx)
rows, cols, values = Mv.data()
Mv = sps.csr_matrix((values, cols, rows))

# check the B
B, Mu = cou.get_inp_opa(cdom=cdom, V=V, NU=NU) 

# get mass mat
v, w = TrialFunction(V), TestFunction(V)
Mv = assemble(inner(w,v)*dx)
rows, cols, values = Mv.data()
Mv = sps.csr_matrix((values, cols, rows))
Bu = spsla.spsolve(Mv,B*np.vstack([0*np.ones((NU,1)),
                                   1*np.ones((NU,1))]))

v = Function(V)
v.vector().set_local(Bu)
plot(v)

# check the C

MyC, My = cou.get_mout_opa(odom=odom, V=V, NY=NY)

exv = Expression(('1', '0'))
testv = interpolate(exv, V)

# plot(testv)

MyCv = MyC*testv.vector().array()
testy = spsla.spsolve(My, MyCv)

# print np.linalg.norm(testy)

ymesh = IntervalMesh(NY-1, odcoo['ymin'], odcoo['ymax'])
Y = FunctionSpace(ymesh, 'CG', 1)
yx = Function(Y)
yy = Function(Y)

# yx.vector().set_local(testy[:NY])
# plot(yx)
# yy.vector().set_local(testy[NY:])
# plot(yy)

## check the regularization of C

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

MyC, My = cou.get_mout_opa(odom=odom, V=femp['V'], NY=NY)
MyC = MyC[:,invinds][:,:]
C = cou.get_regularzd_c(MyC, My, J=stokesmatsc['J'], M=stokesmatsc['M'])

testvi = testv.vector().array()[invinds]
testy = np.dot(C,testvi)


# yx.vector().set_local(testy[:NY])
# plot(yx)
# yy.vector().set_local(testy[NY:])
# plot(yy)

## check if the projection is indeed a projection
os.remove('data/tildeCNY14vdim3042.npy')
Ci = cou.get_regularzd_c(sps.csr_matrix(My*C), My, J=stokesmatsc['J'], M=stokesmatsc['M'])
print np.linalg.norm(np.dot(C,testvi) - np.dot(Ci,testvi))

interactive(True)
