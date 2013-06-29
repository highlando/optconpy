from dolfin import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import linsolv_utils 

def get_inp_opa(cdom=None, NU=8, V=None): 
    """assemble the 'B' matrix

    the findim array representation 
    of the input operator """

    v = TrialFunction(V)

    BX, BY = [], []

    for nbf in range(NU):
        ubf = L2abLinBas(nbf, NU)
        bux = Cast1Dto2D(ubf, cdom, vcomp=0, xcomp=0)
        buy = Cast1Dto2D(ubf, cdom, vcomp=1, xcomp=0)
        bx = inner(v,bux)*dx
        by = inner(v,buy)*dx
        Bx = assemble(bx)
        By = assemble(by)
        Bx = Bx.array()
        By = By.array()
        Bx = Bx.reshape(len(Bx), 1)
        By = By.reshape(len(By), 1)
        BX.append(sps.csc_matrix(By))
        BY.append(sps.csc_matrix(Bx))

    Mu = ubf.massmat()

    return sps.hstack([sps.hstack(BX), sps.hstack(BY)], format='csc'), sps.block_diag([Mu,Mu])

def get_mout_opa(odom=None, NY=8, V=None): 
    """assemble the 'MyC' matrix

    the find an array representation 
    of the output operator 

    the considered output is y(s) = 1/C int_x[1] v(s,x[1]) dx[1]
    it is computed by testing computing my_i = 1/C int y_i v d(x)
    where y_i depends only on x[1] and y_i is zero outside the domain
    of observation. 1/C is the width of the domain of observation
    cf. doku
    """

    v = TestFunction(V)
    domains = CellFunction('uint', V.mesh())
    domains.set_all(0)
    odom.mark(domains, 1)

    dx = Measure('dx')[domains]

    Ci = 1.0/(odom.maxxy[0] - odom.minxy[0])

    YX, YY = [], []

    for nbf in range(NY):
        ybf = L2abLinBas(nbf, NY, a=odom.minxy[1], b=odom.maxxy[1])
        yx = Cast1Dto2D(ybf, odom, vcomp=0, xcomp=1)
        yy = Cast1Dto2D(ybf, odom, vcomp=1, xcomp=1)
        # if nbf == 1:
        #     raise Warning('TODO: debug') 
        # plot(yx, V.mesh())
        yx = Ci*inner(v,yx)*dx(1)
        yy = Ci*inner(v,yy)*dx(1)
        Yx = assemble(yx)
        Yy = assemble(yy)
        Yx = Yx.array()
        Yy = Yy.array()
        Yx = Yx.reshape(1, len(Yx))
        Yy = Yy.reshape(1, len(Yy))
        YX.append(sps.csc_matrix(Yx))
        YY.append(sps.csc_matrix(Yy))

    My = ybf.massmat()

    return sps.vstack([sps.vstack(YX), sps.vstack(YY)], format='csc'), sps.block_diag([My,My])

def app_difffreeproj(v=None, J=None, M=None):
    """apply the regularization (projection to divfree vels)

    i.e. compute v = [I-M^-1*J.T*S^-1*J]v 
    """

    vg = linsolv_utils.app_schurc_inv(M,J, np.atleast_2d(J*v).T)

    vg = spsla.spsolve(M,J.T*vg)

    return v - vg


def get_regularized_c(Ct=None, J=None, Mt=None):
    """apply the regularization (projection to divfree vels)

    i.e. compute rC = C*[I-M^-1*J.T*S^-1*J] as
    rCT = [I - J.T*S.-T*J*M.-T]*C.T
    """

    Nv, NY = Mt.shape[0], Ct.shape[1]
    try:
        rCt = np.load('data/regCNY{0}vdim{1}.npy'.format(NY, Nv))
    except IOError:
        print 'no data/regCNY{0}vdim{1}.npy'.format(NY, Nv)
        MTlu = spsla.splu(Mt)
        auCt = np.zeros(Ct.shape)
        # M.-T*C.T
        for ccol in range(NY):
            auCt[:,ccol] = MTlu.solve(np.array(Ct[:,ccol].todense())[:,0])
        # J*M.-T*C.T
        auCt = J*auCt
        # S.-T*J*M.-T*C.T
        auCt = linsolv_utils.app_schurc_inv(MTlu, J, auCt)
        rCt = Ct - J.T*auCt
        np.save('data/regCNY{0}vdim{1}.npy'.format(NY, Nv), rCt)

    return np.array(rCt)


# Subdomains of Control and Observation
class ContDomain(SubDomain):
    def __init__(self, ddict):
        SubDomain.__init__(self)
        self.minxy = [ddict['xmin'], ddict['ymin']]
        self.maxxy = [ddict['xmax'], ddict['ymax']]
    def inside(self, x, on_boundary):
        epps = 3e-16
        return ( between(x[0], (self.minxy[0]-epps, self.maxxy[0]+epps)) and
                 between(x[1], (self.minxy[1]-epps, self.maxxy[1]+epps)) )

class L2abLinBas():
    """ return the hat function related to the num-th vertex

    from the interval [a=0, b=1] with an equispaced grid
    of N vertices """

    def __init__(self, num, N, a=0.0, b=1.0):
        self.dist = (b-a)/(N-1)
        self.vertex = a + num*self.dist
        self.num, self.N = num, N
        self.a, self.b = a, b

    def evaluate(self, s):
        # print s
        if self.vertex - self.dist <= s <= self.vertex:
            sval = 1.0 - 1.0/self.dist*(self.vertex - s)
        elif self.vertex <= s <= self.vertex + self.dist:
            sval = 1.0 - 1.0/self.dist*(s - self.vertex)
        else:
            sval = 0
        return sval

    def massmat(self):
        """ return the mass matrix 
        """
        mesh = IntervalMesh(self.N-1, self.a, self.b)
        Y = FunctionSpace(mesh, 'CG', 1)
        yv = TestFunction(Y)
        yu = TrialFunction(Y)
        my = yv*yu*dx
        my = assemble(my)
        rows, cols, values = my.data()
        return sps.csr_matrix((values, cols, rows))

class Cast1Dto2D(Expression):
    """ casts a function u defined on [u.a, u.b]

    into the f[comp] of an expression 
    defined on a 2D domain cdom by
    by scaling to fit the xcomp extension
    and simply extruding into the other direction
    """

    def __init__(self, u, cdom, vcomp=0, xcomp=0):
        # control 1D basis function
        self.u = u 
        # domain of control
        self.cdom = cdom
        # component of the value to be set as u(s)
        self.vcomp = vcomp
        # component of x to be considered as s coordinate 
        self.xcomp = xcomp
        # transformation of the intervals [cd.xmin, cd.xmax] -> [a, b]
        # via s = m*x + d
        self.m = (self.u.b - self.u.a)/(cdom.maxxy[self.xcomp] - cdom.minxy[self.xcomp])
        self.d = self.u.b - self.m*cdom.maxxy[self.xcomp]

    def eval(self, value, x):
        value[:] = 0
        if self.cdom.inside(x, False):
            value[self.vcomp] = self.u.evaluate(self.m*x[self.xcomp]+self.d)

    def value_shape(self):
        return (2,)

def get_rightinv(C):
    """compute the rightinverse bmo SVD

    """
    # u, s, vt = spsla.svds(C, k=C.shape[0])
    u, s, vt = np.linalg.svd(C.todense(), full_matrices=0)

    return np.dot(vt.T, np.dot(np.diag(1.0/s), u.T))
