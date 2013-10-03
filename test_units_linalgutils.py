import unittest
import sympy as smp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

# unittests for the helper functions

class TestLinalgUtils(unittest.TestCase):

    def setUp(self):

        self.n = 500
        self.k = 15
        self.A = 30*sps.eye(self.n) + \
                sps.rand(self.n, self.n, format='csr')
        self.U = np.random.randn(self.n, self.k)
        self.V = np.random.randn(self.k, self.n)
        self.Z = np.random.randn(self.n, self.k+2)

    def test_smw_formula(self):
        """check the use of the smw formula

        for the inverse of A-UV"""

        import lin_alg_utils as lau

        # check the branch with direct solves
        AuvInvZ = lau.app_smw_inv(self.A, umat=self.U, vmat=self.V,
                                    rhsa=self.Z, Sinv=None)
        AAinvZ = self.A*AuvInvZ - np.dot(self.U, 
                                    np.dot(self.V, AuvInvZ))
        self.assertTrue(np.allclose(AAinvZ, self.Z))

        #check the branch where A comes as LU
        Alu = spsla.splu(self.A)
        AuvInvZ = lau.app_smw_inv(self.A, umat=self.U, vmat=self.V,
                                    rhsa=self.Z, Sinv=None)
        AAinvZ = self.A*AuvInvZ - np.dot(self.U, 
                                    np.dot(self.V, AuvInvZ))
        self.assertTrue(np.allclose(AAinvZ, self.Z))

    def test_luinv_to_spmat(self):
        """check the application of the inverse 

        of a lu-factored matrix to a sparse mat"""

        import lin_alg_utils as lau

        Alu = spsla.splu(self.A)
        Z = sps.csr_matrix(self.U)
        AinvZ = lau.app_luinv_to_spmat(Alu, Z)

        self.assertTrue(np.allclose(self.U, self.A*AinvZ))

    def test_comp_frobnorm_factored_difference(self):
        """check the computation of the frobenius norm

        """

        import lin_alg_utils as lau

        U = self.U
        Z = self.Z

        # test the branch that returns only the difference
        my_frob_zmu = lau.comp_sqrdfrobnorm_factored_difference(U, Z)
        frob_zmu = np.linalg.norm(np.dot(U, U.T) - np.dot(Z, Z.T),'fro')

        self.assertTrue(np.allclose(frob_zmu*frob_zmu,my_frob_zmu))

        # test the branch that returns difference, norm z1, norm z2
        my_frob_zmu, norm_u, norm_z =  \
                lau.comp_sqrdfrobnorm_factored_difference(U, Z, 
                        ret_sing_norms=True)

        frob_zmu = np.linalg.norm(np.dot(U, U.T) - np.dot(Z, Z.T),'fro')
        frob_u   = np.linalg.norm(np.dot(U, U.T))
        frob_z   = np.linalg.norm(np.dot(Z, Z.T))

        self.assertTrue(np.allclose(frob_zmu*frob_zmu,my_frob_zmu))
        self.assertTrue(np.allclose(norm_u, frob_u**2))
        self.assertTrue(np.allclose(norm_z, frob_z**2))


suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalgUtils)
unittest.TextTestRunner(verbosity=2).run(suite)
