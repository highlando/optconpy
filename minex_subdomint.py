import dolfin
from dolfin import dx, inner

class LocFun(dolfin.Expression):
    """ a locally defined 1D function"""
    def __init__(self, xmin, xmax, fvalue=1):
        self.xmin = xmin
        self.xmax = xmax
        self.fvalue = fvalue
    def eval(self, value, x):
        value[:] = self.fvalue if self.xmin < x[0] < self.xmax else 0
    def value_shape(self):
        return (1,)

class LocFuncDom(dolfin.SubDomain):
    """ domain of definition as subdomain """
    def __init__(self, xmin, xmax):
        dolfin.SubDomain.__init__(self, xmin, xmax)
        self.xmin, self.xmax = xmin, xmax
    def inside(self, x, on_boundary):
        return (dolfin.between(x[0], (self.xmin, self.xmax)))

smin, smax = 0.2, 0.3

mesh = dolfin.IntervalMesh(5, 1, 0)
dolfin.plot(mesh)

Y = dolfin.FunctionSpace(mesh, 'CG', 1)

y = dolfin.Expression('1')
y = dolfin.interpolate(y, Y)

domains = dolfin.CellFunction('uint', Y.mesh())
domains.set_all(0)

ffunc = LocFun(smin, smax)
funcdom = LocFuncDom(smin, smax)
funcdom.mark(domains, 1)

y1 = inner(y, ffunc) * dx
y2 = inner(y, ffunc) * dx(1)

print dolfin.assemble(y1), dolfin.assemble(y2)
