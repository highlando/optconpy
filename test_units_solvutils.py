import unittest
import sympy as smp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

# unittests for the helper functions

class smw_formula(unittest.TestCase):

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

        import linsolv_utils as lsu

        # check the branch with direct solves
        AuvInvZ = lsu.app_smw_inv(self.A, U=self.U, V=self.V,
                                    rhsa=self.Z, Sinv=None)
        AAinvZ = self.A*AuvInvZ - np.dot(self.U, 
                                    np.dot(self.V, AuvInvZ))
        self.assertTrue(np.allclose(AAinvZ, self.Z))

        #check the branch where A comes as LU
        Alu = spsla.splu(self.A)
        AuvInvZ = lsu.app_smw_inv(self.A, U=self.U, V=self.V,
                                    rhsa=self.Z, Sinv=None)
        AAinvZ = self.A*AuvInvZ - np.dot(self.U, 
                                    np.dot(self.V, AuvInvZ))
        self.assertTrue(np.allclose(AAinvZ, self.Z))

    def test_luinv_to_spmat(self):
        """check the application of the inverse 

        of a lu-factored matrix to a sparse mat"""

        import linsolv_utils as lsu

        Alu = spsla.splu(self.A)
        Z = sps.csr_matrix(self.U)
        AinvZ = lsu.app_luinv_to_spmat(Alu, Z)

        self.assertTrue(np.allclose(self.U, self.A*AinvZ))


if __name__ == '__main__':
    unittest.main()
