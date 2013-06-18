from dolfin import *
import numpy as np
import scipy.sparse as sps

def test_ioops():
    N = 12
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "CG", 2)

    NU = 8
    meshu = UnitIntervalMesh(NU)
    U = FunctionSpace(meshu, "CG", 1)

    dood = dict(xmin=0.45,
                xmax=0.55,
                ymin=0.5,
                ymax=0.7)

    # Subdomains of Control and Observation
    class ContDomain(SubDomain):
        def inside(self, x, interior):
            return (between(x[0], (dood['xmin'], dood['xmax'])) and
                    between(x[1], (dood['ymin'], dood['ymax'])))

    domains = CellFunction('uint', mesh)
    domains.set_all(0)

    contdomain = ContDomain()
    contdomain.mark(domains, 1)

    dx = Measure('dx')[domains]

    v = TestFunction(V)
    u = TrialFunction(U)

    u1 = L2abLinBas(2,8)
    xv = np.arange(0,1,0.05)
    vv = np.zeros(1)

    for x in xv:
        u1.eval(vv,np.array([x]))
        print x, vv

class L2abLinBas(Expression):
    def __init__(self, num, N, a=0.0, b=1.0):
        self.dist = (b-a)/(N+1)
        self.vert = num*self.dist
        self.num = num

    def eval(self, value, x):
        if self.vert - self.dist <= x[0] <= self.vert:
            value[0] = 1.0 - 1.0/self.dist*(self.vert - x[0])
        elif self.vert <= x[0] <= self.vert + self.dist:
            value[0] = 1.0 - 1.0/self.dist*(x[0] - self.vert)
        else:
            value[0] = 0

    #def value_shape(self):
    #    return (1,)


if __name__ == '__main__':
	test_ioops()
