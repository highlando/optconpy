import dolfin

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
