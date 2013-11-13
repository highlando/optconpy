from dolfin import *
import numpy as np
import scipy.sparse as sps

parameters.linear_algebra_backend = "uBLAS"

class ContDomain(dolfin.SubDomain):
    """define a subdomain"""
    def __init__(self, ddict):
        dolfin.SubDomain.__init__(self)
        self.minxy = [ddict['xmin'], ddict['ymin']]
        self.maxxy = [ddict['xmax'], ddict['ymax']]
    def inside(self, x, on_boundary):
        return (dolfin.between(x[0], (self.minxy[0], self.maxxy[0]))
                and
                dolfin.between(x[1], (self.minxy[1], self.maxxy[1])))

class CharactFun(dolfin.Expression):
    """ characteristic function of subdomain """
    def __init__(self, subdom):
        self.subdom = subdom
    def eval(self, value, x):
        if self.subdom.inside(x, False):
            value[:] = 1
            print x, value
        else:
            value[:] = 0
    # def value_shape(self):
    #     return (2,)

epsil = 0.0
odcoo = dict(xmin=0.45-epsil,
             xmax=0.55+epsil,
             ymin=0.6-epsil,
             ymax=0.8+epsil)

mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)
plot(mesh)

odom = ContDomain(odcoo)

charfun = CharactFun(odom)
v = TestFunction(V)
u = TrialFunction(V)

MP = assemble(inner(v, u) * charfun * dx ,
                                 form_compiler_parameters={
                                     'quadrature_degree': 3})

rows, cols, values = MP.data()
MPa = sps.dia_matrix(sps.csr_matrix((values, cols, rows)))

checkf = MPa.diagonal()

dofs_on_subd = np.where(checkf > 0)[0]
print dofs_on_subd.shape

basfun = Function(V)
basfun.vector()[dofs_on_subd] = 0.2
basfun.vector()[-2] = 1  # for scaling the others only 
plot(basfun)

interactive(True)
