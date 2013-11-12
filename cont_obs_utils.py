import dolfin
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import lin_alg_utils as lau

from dolfin import dx, inner


def get_inp_opa(cdom=None, NU=8, V=None):
    """dolfin.assemble the 'B' matrix

    the findim array representation
    of the input operator """

    v = dolfin.dolfin.TrialFunction(V)

    BX, BY = [], []

    for nbf in range(NU):
        ubf = L2abLinBas(nbf, NU)
        bux = Cast1Dto2D(ubf, cdom, vcomp=0, xcomp=0)
        buy = Cast1Dto2D(ubf, cdom, vcomp=1, xcomp=0)
        bx = inner(v, bux) * dx
        by = inner(v, buy) * dx
        Bx = dolfin.assemble(bx)
        By = dolfin.assemble(by)
        Bx = Bx.array()
        By = By.array()
        Bx = Bx.reshape(len(Bx), 1)
        By = By.reshape(len(By), 1)
        BX.append(sps.csc_matrix(By))
        BY.append(sps.csc_matrix(Bx))

    Mu = ubf.massmat()

    return (
        sps.hstack([sps.hstack(BX), sps.hstack(BY)],
                   format='csc'), sps.block_diag([Mu, Mu])
    )


def get_mout_opa(odcoo=None, NY=8, V=None, thicken=None):
    """dolfin.assemble the 'MyC' matrix

    the find an array representation
    of the output operator

    the considered output is y(s) = 1/C int_x[1] v(s,x[1]) dx[1]
    it is computed by testing computing my_i = 1/C int y_i v d(x)
    where y_i depends only on x[1] and y_i is zero outside the domain
    of observation. 1/C is the width of the domain of observation
    cf. doku
    """

    if thicken is not None:
        # thickening of the control domain.
        # this is needed since FEniCS integration only
        # considers complete cells
        odcoopl = dict(xmin=odcoo['xmin']-thicken,
                       xmax=odcoo['xmax']+thicken,
                       ymin=odcoo['ymin']-thicken,
                       ymax=odcoo['ymax']+thicken)
    else:
        odcoopl = odcoo

    odom_thick = ContDomain(odcoopl)
    odom = ContDomain(odcoo)

    v = dolfin.TestFunction(V)
    vone = dolfin.Expression(('1', '1'))
    vone = dolfin.interpolate(vone, V)

    domains = dolfin.CellFunction('uint', V.mesh())
    domains.set_all(0)
    odom_thick.mark(domains, 1)

    subdx = dolfin.Measure('dx')[domains]

    # factor to compute the average via \bar u = 1/h \int_0^h u(x) dx
    Ci = 1.0 / (odcoo['xmax'] - odcoo['xmin'])

    YX, YY = [], []

    omega_y = dolfin.RectangleMesh(odcoo['xmin'], odcoo['ymin'],
                                   odcoo['xmax'], odcoo['ymax'],
                                   5, NY-1)

    y_y = dolfin.VectorFunctionSpace(omega_y, 'CG', 1)

    obdom_dofs = extract_dofs_subdomain(V, odom)

    for curdof in obdom_dofs:
        vcur = dolfin.Function(V)
        vcur.vector()[:] = 0
        vcur.vector()[curdof] = 1
        vone_y = dolfin.interpolate(vcur, y_y)

        for nbf in range(NY):
            ybf = L2abLinBas(nbf, NY, a=odcoo['ymin'], b=odcoo['ymax'])
            yx = Cast1Dto2D(ybf, odom, vcomp=0, xcomp=1)
            yy = Cast1Dto2D(ybf, odom, vcomp=1, xcomp=1)

            yxf = Ci * inner(v, yx) * subdx(1)
            yyf = Ci * inner(v, yy) * subdx(1)

            yxone = inner(vone_y, yx) * dx
            yyone = inner(vone, yy) * subdx(1)

            Yx = dolfin.assemble(yxf)
            # ,
            #                      form_compiler_parameters={
            #                          'quadrature_rule': 'canonical',
            #                          'quadrature_degree': 2})
            Yy = dolfin.assemble(yyf)
            # ,
            #                      form_compiler_parameters={
            #                          'quadrature_rule': 'default',
            #                          'quadrature_degree': 2})

            print dolfin.assemble(yxone), dolfin.assemble(yyone)

            Yx = Yx.array()
            Yy = Yy.array()
            Yx = Yx.reshape(1, len(Yx))
            Yy = Yy.reshape(1, len(Yy))
            YX.append(sps.csc_matrix(Yx))
            YY.append(sps.csc_matrix(Yy))

    My = ybf.massmat()

    return (
        sps.vstack([sps.vstack(YX), sps.vstack(YY)],
                   format='csc'), sps.block_diag([My, My])
    )


def app_difffreeproj(v=None, J=None, M=None):
    """apply the regularization (projection to divfree vels)

    i.e. compute v = [I-M^-1*J.T*S^-1*J]v
    """

    vg = lau.app_schurc_inv(M, J, np.atleast_2d(J * v).T)

    vg = spsla.spsolve(M, J.T * vg)

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
            auCt[:, ccol] = MTlu.solve(np.array(Ct[:, ccol].todense())[:, 0])
        # J*M.-T*C.T
        auCt = J * auCt
        # S.-T*J*M.-T*C.T
        auCt = lau.app_schurc_inv(MTlu, J, auCt)
        rCt = Ct - J.T * auCt
        np.save('data/regCNY{0}vdim{1}.npy'.format(NY, Nv), rCt)

    return np.array(rCt)


# Subdomains of Control and Observation
class ContDomain(dolfin.SubDomain):

    def __init__(self, ddict):
        dolfin.SubDomain.__init__(self)
        self.minxy = [ddict['xmin'], ddict['ymin']]
        self.maxxy = [ddict['xmax'], ddict['ymax']]

    def inside(self, x, on_boundary):
        return (dolfin.between(x[0], (self.minxy[0], self.maxxy[0]))
                and
                dolfin.between(x[1], (self.minxy[1], self.maxxy[1])))


class L2abLinBas():
    """ return the hat function related to the num-th vertex

    from the interval [a=0, b=1] with an equispaced grid
    of N vertices """

    def __init__(self, num, N, a=0.0, b=1.0):
        self.dist = (b - a) / (N - 1)
        self.vertex = a + num * self.dist
        self.num, self.N = num, N
        self.a, self.b = a, b

    def evaluate(self, s):
        # print s
        if max(self.a, self.vertex - self.dist) <= s <= self.vertex:
            sval = 1.0 - 1.0 / self.dist * (self.vertex - s)
        elif self.vertex <= s <= min(self.b, self.vertex + self.dist):
            sval = 1.0 - 1.0 / self.dist * (s - self.vertex)
        else:
            sval = 0
        return sval

    def massmat(self):
        """ return the mass matrix
        """
        mesh = dolfin.IntervalMesh(self.N - 1, self.a, self.b)
        Y = dolfin.FunctionSpace(mesh, 'CG', 1)
        yv = dolfin.TestFunction(Y)
        yu = dolfin.TrialFunction(Y)
        my = yv * yu * dx
        my = dolfin.assemble(my)
        rows, cols, values = my.data()
        return sps.csr_matrix((values, cols, rows))


class Cast1Dto2D(dolfin.Expression):
    """ casts a function u defined on [u.a, u.b]

    into the f[comp] of an expression
    defined on a 2D domain cdom by
    by scaling to fit the xcomp extension
    and simply extruding into the other direction
    """

    def __init__(self, u, cdom, vcomp=None, xcomp=0):
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
        self.m = (self.u.b - self.u.a) / \
            (cdom.maxxy[self.xcomp] - cdom.minxy[self.xcomp])
        self.d = self.u.b - self.m * cdom.maxxy[self.xcomp]

    def eval(self, value, x):
        if self.cdom.inside(x, False):
            if self.xcomp is None:
                value[:] = self.u.evaluate(
                    self.m * x[self.xcomp] + self.d)
            else:
                value[:] = 0
                value[self.vcomp] = self.u.evaluate(
                    self.m * x[self.xcomp] + self.d)

    def value_shape(self):
        return (2,)


def get_rightinv(C):
    """compute the rightinverse bmo SVD

    """
    # use numpy routine for dense matrices
    try:
        u, s, vt = np.linalg.svd(np.array(C.todense()), full_matrices=0)
    except AttributeError:
        u, s, vt = np.linalg.svd(C, full_matrices=0)

    return np.dot(vt.T, np.dot(np.diag(1.0 / s), u.T))


def get_vstar(C, ystar, odcoo, NY):

    ystarvec = get_ystarvec(ystar, odcoo, NY)
    Cgeninv = get_rightinv(C)

    return np.dot(Cgeninv, ystarvec)


def get_ystarvec(ystar, odcoo, NY):
    """get the vector of the current target signal

    """
    ymesh = dolfin.IntervalMesh(NY - 1, odcoo['ymin'], odcoo['ymax'])
    Y = dolfin.FunctionSpace(ymesh, 'CG', 1)

    ystarvec = np.zeros((NY * len(ystar), 1))
    for k, ysc in enumerate(ystar):
        cyv = dolfin.interpolate(ysc, Y)
        ystarvec[k * NY:(k + 1) * NY, 0] = cyv.vector().array()

    return ystarvec


def extract_dofs_subdomain(V, subd):
    mesh = V.mesh()
    dofs_of_V = V.dofmap().vertex_to_dof_map(mesh)
    # in 1.3: dofs_of_V = dof_to_vertex_map(V)
    coord_of_dofs = V.mesh().coordinates()
    ncords = coord_of_dofs.shape[0]

    subd_bools = np.zeros(V.dim())
    for inds in dofs_of_V:
        subd_bools[inds] = subd.inside(coord_of_dofs[np.mod(inds, ncords)],
                                       False)

    return np.arange(V.dim())[subd_bools.astype(bool)]
