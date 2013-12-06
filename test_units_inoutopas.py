import unittest
import dolfin
import numpy as np
import scipy.sparse.linalg as spsla

import cont_obs_utils as cou
import lin_alg_utils as lau


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)


# unittests for the input and output operators


class TestInoutOpas(unittest.TestCase):

    # @unittest.skip("for now")
    def test_outopa_workingconfig(self):
        """ The innerproducts that assemble the output operator

        are accurately sampled for this parameter set (NV=25, NY=5)"""

        NV = 25 
        NY = 5

        mesh = dolfin.UnitSquareMesh(NV, NV)
        V = dolfin.VectorFunctionSpace(mesh, "CG", 2)

        exv = dolfin.Expression(('1', '1'))
        testv = dolfin.interpolate(exv, V)

        odcoo = dict(xmin=0.45,
                     xmax=0.55,
                     ymin=0.6,
                     ymax=0.8)

        # check the C
        MyC, My = cou.get_mout_opa(odcoo=odcoo, V=V, NY=NY, NV=NV)

        # signal space
        ymesh = dolfin.IntervalMesh(NY - 1, odcoo['ymin'], odcoo['ymax'])

        Y = dolfin.FunctionSpace(ymesh, 'CG', 1)

        y1 = dolfin.Function(Y)
        y2 = dolfin.Function(Y)

        testvi = testv.vector().array()
        testy = spsla.spsolve(My, MyC * testvi)

        y1 = dolfin.Expression('1')
        y1 = dolfin.interpolate(y1, Y)

        y2 = dolfin.Function(Y)
        y2.vector().set_local(testy[NY:])

        self.assertTrue(dolfin.errornorm(y2, y1) < 1e-14)

    def test_output_opa(self):
        """ test the regularization of the output operator

        """
        import dolfin
        import dolfin_to_nparrays as dtn
        import cont_obs_utils as cou

        from optcont_main import drivcav_fems

        dolfin.parameters.linear_algebra_backend = "uBLAS"

        N = 20
        mesh = dolfin.UnitSquareMesh(N, N)
        V = dolfin.VectorFunctionSpace(mesh, "CG", 2)

        NY = 8

        odcoo = dict(xmin=0.45,
                     xmax=0.55,
                     ymin=0.6,
                     ymax=0.8)

        # get the system matrices
        femp = drivcav_fems(N)
        stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], nu=1)
        # remove the freedom in the pressure
        stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
        stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]
        # reduce the matrices by resolving the BCs
        # (stokesmatsc,
        #  rhsd_stbc,
        #  invinds,
        #  bcinds,
        #  bcvals) = dtn.condense_sysmatsbybcs(stokesmats, femp['diribcs'])

        # check the C
        MyC, My = cou.get_mout_opa(odcoo=odcoo, V=V, NY=NY)
        # exv = dolfin.Expression(('x[1]', 'x[1]'))
        import sympy as smp
        x, y = smp.symbols('x[0], x[1]')
        u_x = x * x * (1 - x) * (1 - x) * 2 * y * (1 - y) * (2 * y - 1)
        u_y = y * y * (1 - y) * (1 - y) * 2 * x * (1 - x) * (1 - 2 * x)
        from sympy.printing import ccode
        exv = dolfin.Expression((ccode(u_x), ccode(u_y)))
        testv = dolfin.interpolate(exv, V)
        # plot(testv)

        # the right inverse to C
        Cplus = cou.get_rightinv(MyC)
        self.assertTrue(np.allclose(np.eye(MyC.shape[0]), MyC * Cplus))

        # MyCc = MyC[:, invinds]

        ptmct = lau.app_prj_via_sadpnt(amat=stokesmats['M'],
                                       jmat=stokesmats['J'],
                                       rhsv=MyC.T,
                                       transposedprj=True)

        testvi = testv.vector().array()
        testvi0 = np.atleast_2d(testv.vector().array()).T
        testvi0 = lau.app_prj_via_sadpnt(amat=stokesmats['M'],
                                         jmat=stokesmats['J'],
                                         rhsv=testvi0)

        # check if divfree part is not zero
        self.assertTrue(np.linalg.norm(testvi0) > 1e-8,
                        msg='maybe nothing wrong, but this shouldnt be zero')

        testyv0 = np.atleast_2d(MyC * testvi0).T
        testry = np.dot(ptmct.T, testvi)

        self.assertTrue(np.allclose(testyv0, testry))


suite = unittest.TestLoader().loadTestsFromTestCase(TestInoutOpas)
unittest.TextTestRunner(verbosity=2).run(suite)
