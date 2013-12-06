import unittest
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import lin_alg_utils as lau

# unittests for the helper functions


class TestLinalgUtils(unittest.TestCase):

    def setUp(self):

        self.n = 1000
        self.k = 30
        self.A = 20 * sps.eye(self.n) + \
            sps.rand(self.n, self.n, format='csr')
        self.U = np.random.randn(self.n, self.k)
        self.V = np.random.randn(self.k, self.n)
        self.Z = np.random.randn(self.n, self.k + 2)
        self.Vsp = sps.rand(self.k, self.n)
        self.J = sps.rand(self.k, self.n)
        self.Jt = sps.rand(self.n, self.k)

    def test_smw_formula(self):
        """check the use of the smw formula

        for the inverse of A-UV"""

        # check the branch with direct solves
        AuvInvZ = lau.app_smw_inv(self.A, umat=self.U, vmat=self.V,
                                  rhsa=self.Z, Sinv=None)
        AAinvZ = self.A * AuvInvZ - np.dot(self.U,
                                           np.dot(self.V, AuvInvZ))
        self.assertTrue(np.allclose(AAinvZ, self.Z))

        # check the branch where A comes as LU
        alusolve = spsla.factorized(self.A)
        AuvInvZ = lau.app_smw_inv(alusolve, umat=self.U, vmat=self.V,
                                  rhsa=self.Z, Sinv=None)
        AAinvZ = self.A * AuvInvZ - np.dot(self.U,
                                           np.dot(self.V, AuvInvZ))
        self.assertTrue(np.allclose(AAinvZ, self.Z))

    def test_smw_formula_spv(self):
        """check the use of the smw formula

        for the inverse of A-UV with v sparse"""

        # check the branch with direct solves
        AuvInvZ = lau.app_smw_inv(self.A, umat=self.U, vmat=self.Vsp,
                                  rhsa=self.Z, Sinv=None)
        AAinvZ = self.A * AuvInvZ - np.dot(self.U, self.Vsp * AuvInvZ)
        self.assertTrue(np.allclose(AAinvZ, self.Z))

        # check the branch where A comes as LU
        alusolve = spsla.factorized(self.A)
        AuvInvZ = lau.app_smw_inv(alusolve, umat=self.U, vmat=self.Vsp,
                                  rhsa=self.Z, Sinv=None)
        AAinvZ = self.A * AuvInvZ - np.dot(self.U, self.Vsp * AuvInvZ)

        self.assertTrue(np.allclose(AAinvZ, self.Z))

    def test_luinv_to_spmat(self):
        """check the application of the inverse

        of a lu-factored matrix to a sparse mat"""

        alusolve = spsla.factorized(self.A)
        Z = sps.csr_matrix(self.U)
        AinvZ = lau.app_luinv_to_spmat(alusolve, Z)

        self.assertTrue(np.allclose(self.U, self.A * AinvZ))

    def test_solve_proj_sadpnt_smw(self):
        """check the sadpnt solver"""

        umat, vmat, k, n = self.U, self.V, self.k, self.n

        # self.Jt = self.J.T
        # check the formula
        AuvInvZ = lau.solve_sadpnt_smw(amat=self.A, jmat=self.J, rhsv=self.Z,
                                       jmatT=self.Jt, umat=self.U, vmat=self.V)

        sysm1 = sps.hstack([self.A, self.Jt], format='csr')
        sysm2 = sps.hstack([self.J, sps.csr_matrix((k, k))], format='csr')
        mata = sps.vstack([sysm1, sysm2], format='csr')

        umate = np.vstack([umat, np.zeros((k, umat.shape[1]))])
        vmate = np.hstack([vmat, np.zeros((vmat.shape[0], k))])
        ze = np.vstack([self.Z, np.zeros((k, self.Z.shape[1]))])

        AAinvZ = mata * AuvInvZ - np.dot(umate, np.dot(vmate, AuvInvZ))

        # likely to fail because of ill conditioned rand mats
        self.assertTrue(np.allclose(AAinvZ, ze),
                        msg='likely to fail because of ill cond')

    def test_sadpnt_smw(self):
        """check the sadpnt as projection"""

        umat, vmat, k, n = self.U, self.V, self.k, self.n

        # check whether it is a projection
        AuvInvZ = lau.solve_sadpnt_smw(amat=self.A, jmat=self.J, rhsv=self.Z,
                                       jmatT=self.Jt, umat=self.U, vmat=self.V)

        auvAUVinv = self.A * AuvInvZ[:n, :] - \
            lau.comp_uvz_spdns(self.U, self.V, AuvInvZ[:n, :])

        AuvInv2Z = lau.solve_sadpnt_smw(amat=self.A, jmat=self.J, rhsv=auvAUVinv,
                                        jmatT=self.Jt,
                                        umat=self.U, vmat=self.V)

        self.assertTrue(np.allclose(AuvInvZ[:n, :], AuvInv2Z[:n, :]),
                        msg='likely to fail because of ill cond')

        prjz = lau.app_prj_via_sadpnt(amat=self.A, jmat=self.J, 
                                      rhsv=self.Z, jmatT=self.Jt)
        prprjz = lau.app_prj_via_sadpnt(amat=self.A, jmat=self.J,   
                                        rhsv=self.Z, jmatT=self.Jt)

        # check projector
        self.assertTrue(np.allclose(prprjz, prjz))

        # onto kernel J
        self.assertTrue(np.linalg.norm(prjz) > 1e-8)
        self.assertTrue(np.linalg.norm(self.J*prjz)/np.linalg.norm(prjz)
                        < 1e-6)

        # check transpose
        idmat = np.eye(n)
        prj = lau.app_prj_via_sadpnt(amat=self.A, jmat=self.J, 
                                     rhsv=idmat, jmatT=self.Jt)

        prjT = lau.app_prj_via_sadpnt(amat=self.A, jmat=self.J, 
                                      rhsv=idmat, jmatT=self.Jt,
                                      transposedprj=True)

        self.assertTrue(np.allclose(prj, prjT.T))


    def test_comp_frobnorm_factored_difference(self):
        """check the computation of the frobenius norm

        """

        U = self.U
        Z = self.Z

        # test the branch that returns only the difference
        my_frob_zmu = lau.comp_sqfnrm_factrd_diff(U, Z)
        frob_zmu = np.linalg.norm(np.dot(U, U.T) - np.dot(Z, Z.T), 'fro')

        self.assertTrue(np.allclose(frob_zmu * frob_zmu, my_frob_zmu))

        # test the branch that returns difference, norm z1, norm z2
        my_frob_zmu, norm_u, norm_z =  \
            lau.comp_sqfnrm_factrd_diff(U, Z, ret_sing_norms=True)

        frob_zmu = np.linalg.norm(np.dot(U, U.T) - np.dot(Z, Z.T), 'fro')
        frob_u = np.linalg.norm(np.dot(U, U.T))
        frob_z = np.linalg.norm(np.dot(Z, Z.T))

        self.assertTrue(np.allclose(frob_zmu * frob_zmu, my_frob_zmu))
        self.assertTrue(np.allclose(norm_u, frob_u ** 2))
        self.assertTrue(np.allclose(norm_z, frob_z ** 2))


suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgUtils)
unittest.TextTestRunner(verbosity=2).run(suite)
