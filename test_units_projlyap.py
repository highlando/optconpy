import unittest

import sympy as smp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

# unittests for the helper functions

class TestProjLyap(unittest.TestCase):

    def setUp(self):
        self.Nv = 200
        self.Np = 150
        self.Ny = 30
        self.adisteps = 150

        # -F, M spd -- coefficient matrices
        self.F = -sps.eye(self.Nv) - sps.rand(self.Nv, self.Nv)*sps.rand(self.Nv, self.Nv) 
        self.M = sps.eye(self.Nv) + sps.rand(self.Nv, self.Nv)*sps.rand(self.Nv, self.Nv) 
        try:
            self.Mlu = spsla.splu(self.M)
        except RuntimeError:
            print 'M is not full rank'

        # right-handside: C= -W*W.T
        self.W = np.random.randn(self.Nv, self.Ny)

        # we need J sparse and of full rank
        for auxk in range(5):
            try:
                self.J = sps.rand(self.Np, self.Nv, density=0.03, format='csr')
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

        Z = pru.solve_proj_lyap_stein(At=self.F.T, Mt=self.M.T, 
                                        J=self.J, W=self.W,
                                        nadisteps=self.adisteps)

        MinvJt = lau.app_luinv_to_spmat(self.Mlu, self.J.T)
        Sinv = np.linalg.inv(self.J*MinvJt)
        P = np.eye(self.Nv)-np.dot(MinvJt,Sinv*self.J)

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

        U = 1e-4*self.W
        V = self.W.T

        Z = pru.solve_proj_lyap_stein(At=self.F.T, Mt=self.M.T, 
                                        ut=U.T, vt=V.T, 
                                        J=self.J, W=self.W,
                                        nadisteps=self.adisteps)

        uvst = sps.csr_matrix(np.dot(V.T,U.T))
        Z2 = pru.solve_proj_lyap_stein(At=self.F.T-uvst, Mt=self.M.T, 
                                        J=self.J, W=self.W,
                                        nadisteps=self.adisteps)

        print 'this should be 0={0}'.format(np.linalg.norm(Z-Z2))

        MinvJt = lau.app_luinv_to_spmat(self.Mlu, self.J.T)
        Sinv = np.linalg.inv(self.J*MinvJt)
        P = np.eye(self.Nv)-np.dot(MinvJt,Sinv*self.J)

        XM = np.dot(Z,Z.T*self.M)
        MtXM = self.M.T*XM

        FtUVXM = self.F.T*np.dot(Z,Z.T)*self.M - np.dot(U, np.dot(V, XM))
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
