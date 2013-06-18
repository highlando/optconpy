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

    xv = np.arange(dood['xmin'], dood['xmax'], 
                    (dood['xmax']-dood['xmin'])/10)

    vv = np.zeros(1)
    bvv = np.zeros(1)

    u1 = L2abLinBas(1,4)
    Bu = Inp2Rhs(u1, dood['xmin'], dood['xmax'])

    for x in xv:
        u1.evaluate(vv, np.array([x]))
        Bu.eval(bvv, np.array([x]))
        print x, vv, bvv

    return u1, Bu

class L2abLinBas():
    def __init__(self, num, N, a=0.0, b=1.0):
        self.dist = (b-a)/(N+1)
        self.vertex = num*self.dist
        self.num, self.N = num, N
        self.a, self.b = a, b

    def evaluate(self, value, x):
        if self.vertex - self.dist <= x[0] <= self.vertex:
            value[0] = 1.0 - 1.0/self.dist*(self.vertex - x[0])
        elif self.vertex <= x[0] <= self.vertex + self.dist:
            value[0] = 1.0 - 1.0/self.dist*(x[0] - self.vertex)
        else:
            value[0] = 0

class Inp2Rhs(Expression):
    """ map the control defined on [u.a, u.b]

    into the domain of control : [cda, cdb] """
    def __init__(self, u, cda, cdb):
        self.u = u 
        self.cda, self.cdb = cda, cdb

    def eval(self, value, x):
        if x[0] < self.cda or self.cdb < x[0] :
            raise UserWarning('x value outside domain of control')
        # transformation of the intervals [cda, cdb] -> [a, b]
        # via xn = m*x + d
        m = (self.u.b - self.u.a)/(self.cdb - self.cda)
        d = self.u.b - m*self.cdb

        self.u.evaluate(value, m*x+d)


if __name__ == '__main__':
	test_ioops()
