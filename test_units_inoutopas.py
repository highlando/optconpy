import unittest
import dolfin
import numpy as np
import cont_obs_utils as cou
import scipy.sparse.linalg as spsla

# unittests for the input and output operators


class TestInoutOpas(unittest.TestCase):

    def test_outopa_workingconfig(self):
        """ The innerproducts that assemble the output operator

        are accurately sampled for this parameter set"""

        NV = 25
        NY = 9

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
        import scipy.sparse.linalg as spsla
        import dolfin_to_nparrays as dtn
        import cont_obs_utils as cou

        from optcont_main import drivcav_fems

        dolfin.parameters.linear_algebra_backend = "uBLAS"

        N = 24
        mesh = dolfin.UnitSquareMesh(N, N)
        V = dolfin.VectorFunctionSpace(mesh, "CG", 2)

        NY = 7

        odcoo = dict(xmin=0.45,
                     xmax=0.55,
                     ymin=0.6,
                     ymax=0.8)

        # get the system matrices
        femp = drivcav_fems(N)
        stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], 1)
        # remove the freedom in the pressure
        stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
        stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]
        # reduce the matrices by resolving the BCs
        (stokesmatsc,
         rhsd_stbc,
         invinds,
         bcinds,
         bcvals) = dtn.condense_sysmatsbybcs(stokesmats, femp['diribcs'])

        # check the C
        MyC, My = cou.get_mout_opa(odcoo=odcoo, V=V, NY=NY)
        # exv = dolfin.Expression(('x[1]', 'x[1]'))
        import sympy as smp
        x, y = smp.symbols('x[0], x[1]')
        u_x = x*x*(1-x)*(1-x)*2*y*(1-y)*(2*y-1)
        u_y = y*y*(1-y)*(1-y)*2*x*(1-x)*(1-2*x)
        from sympy.printing import ccode
        exv = dolfin.Expression((ccode(u_x), ccode(u_y)))
        testv = dolfin.interpolate(exv, V)
        # plot(testv)

        # the right inverse to C
        Cplus = cou.get_rightinv(MyC)
        self.assertTrue(np.allclose(np.eye(MyC.shape[0]), MyC * Cplus))

        # MyCv = MyC*testv.vector().array()
        # testy = spsla.spsolve(My, MyCv)
        # print np.linalg.norm(testy)

        # ymesh = dolfin.IntervalMesh(NY-1, odcoo['ymin'], odcoo['ymax'])
        # Y = dolfin.FunctionSpace(ymesh, 'CG', 1)
        # y1 = dolfin.Function(Y)
        # y2 = dolfin.Function(Y)
        # y3 = dolfin.Function(Y)
        # y4 = dolfin.Function(Y)
        # y5 = dolfin.Function(Y)
        # y6 = dolfin.Function(Y)

        # interactive(True)
        # y1.vector().set_local(testy[:NY])
        # plot(y1)
        # y2.vector().set_local(testy[NY:])
        # plot(y2)

        # check the regularization of C
        rC = cou.get_regularized_c(
            MyC.T,
            J=stokesmats['J'],
            Mt=stokesmats['M']).T

        testvi = testv.vector().array()
        testvi0 = cou.app_difffreeproj(
            M=stokesmats['M'],
            J=stokesmats['J'],
            v=testvi)

        # check if divfree part is not zero
        self.assertTrue(np.linalg.norm(testvi0) > 1e-8)

        testyv0 = spsla.spsolve(My, MyC * testvi0)
        testry = spsla.spsolve(My, np.dot(rC, testvi))

        self.assertTrue(np.allclose(testyv0, testry))

        # testyg = spsla.spsolve(My, MyC*(testvi-testvi0))
        # y3.vector().set_local(testry[NY:])
        # plot(y3, title='rCv')
        # y4.vector().set_local(testyv0[NY:])
        # plot(y4, title='Cv0')
        # y5.vector().set_local(testyg[NY:])
        # plot(y5, title='Cvg')

        # check if the projection is indeed a projection
        # os.remove('data/regCNY14vdim3042.npy')
        # Ci = cou.get_regularized_c(sps.csr_matrix(My*C).T,
        #                             J=stokesmatsc['J'],
        #                             Mt=stokesmatsc['M']).T
        # print np.linalg.norm(np.dot(C,testvi) - np.dot(Ci,testvi))

suite = unittest.TestLoader().loadTestsFromTestCase(TestInoutOpas)
unittest.TextTestRunner(verbosity=2).run(suite)
