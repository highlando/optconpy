from dolfin import *
import numpy as np
import scipy.sparse as sps

def get_inp_opa(cdom=None, NU=8, V=None): 
    """assemble the 'B' matrix

    the findim array representation 
    of the input operator """

    v = TrialFunction(V)

    BX, BY = [], []

    for nbf in range(NU+1):
        ubf = L2abLinBas(nbf, NU)
        bux = Inp2Rhs2D(ubf, cdom, vcomp=0, xcomp=0)
        buy = Inp2Rhs2D(ubf, cdom, vcomp=1, xcomp=0)

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

    return sps.hstack([sps.hstack(BX), sps.hstack(BY)])


# Subdomains of Control and Observation
class ContDomain(SubDomain):
    def __init__(self, ddict):
        SubDomain.__init__(self)
        self.xmin, self.xmax = ddict['xmin'], ddict['xmax']
        self.ymin, self.ymax = ddict['ymin'], ddict['ymax']
    def inside(self, x, on_boundary):
        return ( between(x[0], (self.xmin, self.xmax)) and
                 between(x[1], (self.ymin, self.ymax)) )

class L2abLinBas():
    """ return the hat function related to the num-th vertex

    from the interval [a=0, b=1] with an equispaced grid
    of N+1 inner vertices """

    def __init__(self, num, N, a=0.0, b=1.0):
        self.dist = (b-a)/N
        self.vertex = num*self.dist
        self.num, self.N = num, N
        self.a, self.b = a, b

    def evaluate(self, s):
        if self.vertex - self.dist <= s <= self.vertex:
            sval = 1.0 - 1.0/self.dist*(self.vertex - s)
        elif self.vertex <= s <= self.vertex + self.dist:
            sval = 1.0 - 1.0/self.dist*(s - self.vertex)
        else:
            sval = 0
        return sval

class Inp2Rhs2D(Expression):
    """ map the control defined on [u.a, u.b]

    into the f[comp] of an expression 
    defined on the control domain cdom
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
        self.m = (self.u.b - self.u.a)/(cdom.xmax - cdom.xmin)
        self.d = self.u.b - self.m*cdom.xmax

    def eval(self, value, x):
        value[:] = 0
        if self.cdom.inside(x, False):
            value[self.vcomp] = self.u.evaluate(self.m*x[self.xcomp]+self.d)

    def value_shape(self):
        return (2,)

