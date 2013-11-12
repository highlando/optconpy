import dolfin

class LocFun(dolfin.Expression):
    """ a locally defined constant function"""
    def __init__(self, xmin, xmax, fvalue=1):
        self.xmin = xmin
        self.xmax = xmax
        self.fvalue = fvalue
    def eval(self, value, x):
        value[:] = self.fvalue if self.xmin < x[0] < self.xmax else 0

class LocFuncDom(dolfin.SubDomain):
    """ domain of definition as subdomain """
    def __init__(self, xmin, xmax):
        dolfin.SubDomain.__init__(self)
        self.xmin, self.xmax = xmin, xmax
    def inside(self, x, on_boundary):
        return (dolfin.between(x[0], (self.xmin, self.xmax)))

# extension of the domain where ffunc is not 0
smin, smax = 0.1, 0.3

mesh = dolfin.IntervalMesh(5, 0, 1)
# dolfin.plot(mesh)

Y = dolfin.FunctionSpace(mesh, 'CG', 1)

y = dolfin.Expression('1')
y = dolfin.interpolate(y, Y)
# dolfin.plot(y)

domains = dolfin.CellFunction('uint', Y.mesh())
domains.set_all(0)

# the support of ffunc as subdomain
ffunc = LocFun(smin, smax)
funcdom = LocFuncDom(smin, smax)
funcdom.mark(domains, 1)

dx = dolfin.Measure('dx')[domains]

# dolfin.plot(ffunc, mesh)
# dolfin.interactive(True)

# ffunc integrated over the whole domain
y1 = dolfin.inner(y, ffunc) * dolfin.dx
# ffunc integrated on over its support
y2 = dolfin.inner(y, ffunc) * dx(1)

print dolfin.assemble(y1), dolfin.assemble(y2)
