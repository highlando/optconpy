import unittest

import sympy as smp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

# unittests for the helper functions

class TestProjLyap(unittest.TestCase):

    def setUp(self):
        self.NV = 200
        self.NP = 150
        self.NY = 3
        self.adisteps = 120

        # -F, M spd -- coefficient matrices
        self.F = -sps.eye(self.NV) - sps.rand(self.NV, self.NV)*sps.rand(self.NV, self.NV) 
        self.M = sps.eye(self.NV) + sps.rand(self.NV, self.NV)*sps.rand(self.NV, self.NV) 
        try:
            self.Mlu = spsla.splu(self.M)
        except RuntimeError:
            print 'M is not full rank'

        # right-handside: C= -W*W.T
        self.W = np.random.randn(self.NV, self.NY)

        # smw formula Asmw = A - UV
        self.U = 1e-4*np.random.randn(self.NV, self.NY)
        self.V = np.random.randn(self.NY, self.NV)

        # we need J sparse and of full rank
        for auxk in range(5):
            try:
                self.J = sps.rand(self.NP, self.NV, density=0.03, format='csr')
                spsla.splu(self.J*self.J.T)
                break
            except RuntimeError:
                print 'J not full row-rank.. I make another try'
        try:
            spsla.splu(self.J*self.J.T)
        except RuntimeError:
            raise Warning('Fail: J is not full rank')

    def test_proj_lyap_sol(self):
        """check the solution of the projected lyap eqn 
        
        via ADI iteration"""
        import lin_alg_utils as lau
        import proj_ric_utils as pru

        Z = pru.solve_proj_lyap_stein(A=self.F, M=self.M, 
                                        J=self.J, W=self.W,
                                        nadisteps=self.adisteps)

        MinvJt = lau.app_luinv_to_spmat(self.Mlu, self.J.T)
        Sinv = np.linalg.inv(self.J*MinvJt)
        P = np.eye(self.NV)-np.dot(MinvJt,Sinv*self.J)

        MtXM = self.M.T*np.dot(Z,Z.T)*self.M

        FtXM = self.F.T*np.dot(Z,Z.T)*self.M
        PtW = np.dot(P.T,self.W)
        ProjRes = np.dot(P.T, np.dot(FtXM, P)) + \
                np.dot( np.dot(P.T, FtXM.T), P) + \
                np.dot(PtW,PtW.T)

        print np.linalg.norm(MtXM)

        self.assertTrue(np.allclose(MtXM,np.dot(P.T,np.dot(MtXM,P))))

        self.assertTrue(np.linalg.norm(ProjRes)/np.linalg.norm(MtXM)
                            < 1e-4 )

    def test_proj_lyap_smw_sol(self):
        """check the solution of the projected lyap eqn 
        
        via ADI iteration"""
        import lin_alg_utils as lau
        import proj_ric_utils as pru

        U = self.U
        V = self.V
        print np.linalg.norm(U)

        Z = pru.solve_proj_lyap_stein(A=self.F, M=self.M, 
                                        umat=U, vmat=V, 
                                        J=self.J, W=self.W,
                                        nadisteps=self.adisteps)

        uvs = sps.csr_matrix(np.dot(U,V))
        Z2 = pru.solve_proj_lyap_stein(A=self.F-uvs, M=self.M, 
                                        J=self.J, W=self.W,
                                        nadisteps=self.adisteps)

        self.assertTrue(np.allclose(Z,Z2))

        MinvJt = lau.app_luinv_to_spmat(self.Mlu, self.J.T)
        Sinv = np.linalg.inv(self.J*MinvJt)
        P = np.eye(self.NV)-np.dot(MinvJt,Sinv*self.J)

        XM = np.dot(Z,Z.T*self.M)
        MtXM = self.M.T*XM

        FtUVXM = self.F.T*np.dot(Z,Z.T)*self.M \
                            - np.dot(V.T, np.dot(U.T, XM))

        PtW = np.dot(P.T,self.W)
        ProjRes = np.dot(P.T, np.dot(FtUVXM, P)) + \
                np.dot( np.dot(P.T, FtUVXM.T), P) + \
                np.dot(PtW,PtW.T)

        print np.linalg.norm(MtXM)

        self.assertTrue(np.allclose(MtXM,np.dot(P.T,np.dot(MtXM,P))))

        self.assertTrue(np.linalg.norm(ProjRes)/np.linalg.norm(MtXM)
                            < 1e-4 )

    def test_proj_alg_ric_sol(self):
        """check the sol of the projected alg. Riccati Eqn

        via Newton ADI"""
        

suite = unittest.TestLoader().loadTestsFromTestCase(TestProjLyap)
unittest.TextTestRunner(verbosity=2).run(suite)
