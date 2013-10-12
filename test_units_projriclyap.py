import unittest

import sympy as smp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import lin_alg_utils as lau

import proj_ric_utils as pru

#### unittests for the helper functions

verbose = False

class TestProjLyap(unittest.TestCase):

    def setUp(self):
        self.NV = 100
        self.NP = 20
        self.NY = 5

        self.nwtn_adi_dict = dict(
                            adi_max_steps=120,
                            adi_newZ_reltol=1e-8,
                            nwtn_max_steps=24,
                            nwtn_upd_reltol=4e-8,
                            nwtn_upd_abstol=4e-8,
                            verbose=verbose
                                    )

        # -F, M spd -- coefficient matrices
        self.F = -sps.eye(self.NV) - \
                sps.rand(self.NV, self.NV)*sps.rand(self.NV, self.NV) 
        self.M = sps.eye(self.NV) + \
                sps.rand(self.NV, self.NV)*sps.rand(self.NV, self.NV) 
        try:
            self.Mlu = spsla.splu(self.M)
        except RuntimeError:
            print 'M is not full rank'

        # bmatrix that appears in the nonliner ric term X*B*B.T*X
        self.bmat = np.random.randn(self.NV, self.NY+3)

        # right-handside: C= -W*W.T
        self.W = np.random.randn(self.NV, self.NY)

        # smw formula Asmw = A - UV
        self.U = 1e-4*np.random.randn(self.NV, self.NY)
        self.V = np.random.randn(self.NY, self.NV)
        self.uvs = sps.csr_matrix(np.dot(self.U, self.V))

        # initial value for newton adi
        self.Z0 = np.random.randn(self.NV, self.NY)

        # we need J sparse and of full rank
        for auxk in range(10):
            try:
                self.J = sps.rand(self.NP, self.NV, 
                        density=0.03, format='csr')
                spsla.splu(self.J*self.J.T)
                break
            except RuntimeError:
                if verbose:
                    print 'J not full row-rank.. I make another try'
        try:
            spsla.splu(self.J*self.J.T)
        except RuntimeError:
            raise Warning('Fail: J is not full rank')

        # the Leray projector
        MinvJt = lau.app_luinv_to_spmat(self.Mlu, self.J.T)
        Sinv = np.linalg.inv(self.J*MinvJt)
        self.P = np.eye(self.NV)-np.dot(MinvJt,Sinv*self.J)


    def test_proj_lyap_sol(self):
        """check the solution of the projected lyap eqn 
        
        via ADI iteration"""

        Z = pru.solve_proj_lyap_stein(A=self.F, M=self.M, 
                                    umat=self.U, vmat=self.V, 
                                    J=self.J, W=self.W,
                                    adi_dict=self.nwtn_adi_dict)['zfac']

        MtXM = self.M.T*np.dot(Z,Z.T)*self.M
        FtXM = (self.F.T-self.uvs.T)*np.dot(Z,Z.T)*self.M

        PtW = np.dot(self.P.T,self.W)

        ProjRes = np.dot(self.P.T, np.dot(FtXM, self.P)) + \
                np.dot( np.dot(self.P.T, FtXM.T), self.P) + \
                np.dot(PtW,PtW.T)

## TEST: result is 'projected'
        self.assertTrue(np.allclose(MtXM,
                                np.dot(self.P.T,np.dot(MtXM,self.P))))

## TEST: check projected residual
        self.assertTrue(np.linalg.norm(ProjRes)/np.linalg.norm(MtXM)
                            < 1e-8 )

    def test_proj_lyap_smw_transposeflag(self):
        """check the solution of the projected lyap eqn 
        
        via ADI iteration"""

        U = self.U
        V = self.V

        Z = pru.solve_proj_lyap_stein(A=self.F, M=self.M, 
                                    umat=U, vmat=V, 
                                    J=self.J, W=self.W,
                                    adi_dict=self.nwtn_adi_dict)['zfac']

        Z2 = pru.solve_proj_lyap_stein(A=self.F-self.uvs, M=self.M, 
                                    J=self.J, W=self.W,
                                    adi_dict=self.nwtn_adi_dict)['zfac']

        Z3 = pru.solve_proj_lyap_stein(A=self.F.T-self.uvs.T, M=self.M.T, 
                                    J=self.J, W=self.W,
                                    adi_dict=self.nwtn_adi_dict,
                                    transposed=True)['zfac']

        Z4 = pru.solve_proj_lyap_stein(A=self.F.T, M=self.M.T, 
                                    J=self.J, W=self.W,
                                    umat=U, vmat=V, 
                                    adi_dict=self.nwtn_adi_dict,
                                    transposed=True)['zfac']

## TEST: {smw} x {transposed}
        self.assertTrue(np.allclose(Z,Z2))
        self.assertTrue(np.allclose(Z2,Z3))
        self.assertTrue(np.allclose(Z3,Z4))


    def test_proj_alg_ric_sol(self):
        """check the sol of the projected alg. Riccati Eqn

        via Newton ADI"""
        Z = pru.proj_alg_ric_newtonadi(mmat=self.M, fmat=self.F, 
                               jmat=self.J, bmat=self.bmat, 
                               wmat=self.W, z0=self.bmat, 
                               nwtn_adi_dict=self.nwtn_adi_dict)['zfac']

        MtXM = self.M.T*np.dot(Z,Z.T)*self.M
        MtXb = self.M.T*np.dot(np.dot(Z, Z.T), self.bmat)

        FtXM = self.F.T*np.dot(Z,Z.T)*self.M
        PtW = np.dot(self.P.T,self.W)


        ProjRes = np.dot(self.P.T, np.dot(FtXM, self.P)) + \
                np.dot(np.dot(self.P.T, FtXM.T), self.P) -\
                np.dot(MtXb, MtXb.T) + \
                np.dot(PtW,PtW.T) 

## TEST: result is 'projected' - riccati sol
        self.assertTrue(np.allclose(MtXM,
                                np.dot(self.P.T,np.dot(MtXM,self.P))))
        
## TEST: check projected residual - riccati sol
        self.assertTrue(np.linalg.norm(ProjRes)/np.linalg.norm(MtXM)
                            < 1e-7 )
        

suite = unittest.TestLoader().loadTestsFromTestCase(TestProjLyap)
unittest.TextTestRunner(verbosity=2).run(suite)