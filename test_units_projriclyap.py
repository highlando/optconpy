import unittest

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import lin_alg_utils as lau

import proj_ric_utils as pru
# unittests for the helper functions


class TestProjLyap(unittest.TestCase):

    def setUp(self):
        self.NV = 200
        self.NP = 40
        self.NY = 5
        self.NU = self.NY+3
        self.verbose = True
        self.compn = 15  # factor for comp Z ~~> compn*W.shape[1]

        self.nwtn_adi_dict = dict(adi_max_steps=300,
                                  adi_newZ_reltol=1e-11,
                                  nwtn_max_steps=24,
                                  nwtn_upd_reltol=4e-7,
                                  nwtn_upd_abstol=4e-7,
                                  full_upd_norm_check=True,
                                  verbose=self.verbose)

        # -F, M spd -- coefficient matrices
        self.F = -sps.eye(self.NV) - \
            sps.rand(self.NV, self.NV) * sps.rand(self.NV, self.NV)
        self.M = sps.eye(self.NV) + \
            sps.rand(self.NV, self.NV) * sps.rand(self.NV, self.NV)
        try:
            self.Mlu = spsla.factorized(self.M.tocsc())
        except RuntimeError:
            print 'M is not full rank'

        # bmatrix that appears in the nonliner ric term X*B*B.T*X
        self.bmat = np.random.randn(self.NV, self.NU)

        # right-handside: C= -W*W.T
        self.W = np.random.randn(self.NV, self.NY)

        # smw formula Asmw = A - UV
        self.U = 1e-4 * np.random.randn(self.NV, self.NY)
        self.Usp = 1e-4 * sps.rand(self.NV, self.NY)
        self.V = np.random.randn(self.NY, self.NV)
        self.uvs = sps.csr_matrix(np.dot(self.U, self.V))
        self.uvssp = sps.csr_matrix(self.Usp * self.V)

        # initial value for newton adi
        self.Z0 = np.random.randn(self.NV, self.NY)

        # we need J sparse and of full rank
        for auxk in range(10):
            try:
                self.J = sps.rand(self.NP, self.NV,
                                  density=0.03, format='csr')
                spsla.splu((self.J * self.J.T).tocsc())
                break
            except RuntimeError:
                if self.verbose:
                    print 'J not full row-rank.. I make another try'
        try:
            spsla.splu((self.J * self.J.T).tocsc())
        except RuntimeError:
            raise Warning('Fail: J is not full rank')

        # the Leray projector
        MinvJt = lau.app_luinv_to_spmat(self.Mlu, self.J.T)
        Sinv = np.linalg.inv(self.J * MinvJt)
        self.P = np.eye(self.NV) - np.dot(MinvJt, Sinv * self.J)

    def test_proj_lyap_sol(self):
        """check the solution of the projected lyap eqn

        via ADI iteration"""

        Z = pru.solve_proj_lyap_stein(A=self.F, M=self.M,
                                      umat=self.U, vmat=self.V,
                                      J=self.J, W=self.W,
                                      adi_dict=self.nwtn_adi_dict)['zfac']

        MtXM = self.M.T * np.dot(Z, Z.T) * self.M
        FtXM = (self.F.T - self.uvs.T) * np.dot(Z, Z.T) * self.M

        PtW = np.dot(self.P.T, self.W)

        ProjRes = np.dot(self.P.T, np.dot(FtXM, self.P)) + \
            np.dot(np.dot(self.P.T, FtXM.T), self.P) + \
            np.dot(PtW, PtW.T)

# TEST: result is 'projected'
        self.assertTrue(np.allclose(MtXM,
                                    np.dot(self.P.T, np.dot(MtXM, self.P))))

# TEST: check projected residual
        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)
                        < 1e-8)

    def test_proj_lyap_sol_sparseu(self):
        """check the solution of the projected lyap eqn

        via ADI iteration"""

        Z = pru.solve_proj_lyap_stein(A=self.F, M=self.M,
                                      umat=self.Usp, vmat=self.V,
                                      J=self.J, W=self.W,
                                      adi_dict=self.nwtn_adi_dict)['zfac']

        MtXM = self.M.T * np.dot(Z, Z.T) * self.M
        FtXM = (self.F.T - self.uvssp.T) * np.dot(Z, Z.T) * self.M

        PtW = np.dot(self.P.T, self.W)

        ProjRes = np.dot(self.P.T, np.dot(FtXM, self.P)) + \
            np.dot(np.dot(self.P.T, FtXM.T), self.P) + \
            np.dot(PtW, PtW.T)

# TEST: result is 'projected'
        self.assertTrue(np.allclose(MtXM,
                                    np.dot(self.P.T, np.dot(MtXM, self.P))))

# TEST: check projected residual
        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)
                        < 1e-8)

    def test_proj_lyap_smw_transposeflag(self):
        """check the solution of the projected lyap eqn

        via ADI iteration"""

        U = self.U
        V = self.V

        Z = pru.solve_proj_lyap_stein(A=self.F, M=self.M,
                                      umat=U, vmat=V,
                                      J=self.J, W=self.W,
                                      adi_dict=self.nwtn_adi_dict)['zfac']

        Z2 = pru.solve_proj_lyap_stein(A=self.F - self.uvs, M=self.M,
                                       J=self.J, W=self.W,
                                       adi_dict=self.nwtn_adi_dict)['zfac']

        Z3 = pru.solve_proj_lyap_stein(A=self.F.T - self.uvs.T, M=self.M.T,
                                       J=self.J, W=self.W,
                                       adi_dict=self.nwtn_adi_dict,
                                       transposed=True)['zfac']

        Z4 = pru.solve_proj_lyap_stein(A=self.F.T, M=self.M.T,
                                       J=self.J, W=self.W,
                                       umat=U, vmat=V,
                                       adi_dict=self.nwtn_adi_dict,
                                       transposed=True)['zfac']

# TEST: {smw} x {transposed}
        self.assertTrue(np.allclose(Z, Z2))
        self.assertTrue(np.allclose(Z2, Z3))
        self.assertTrue(np.allclose(Z3, Z4))
        self.assertTrue(np.allclose(Z, Z4))

    def test_proj_alg_ric_sol(self):
        """check the sol of the projected alg. Riccati Eqn

        via Newton ADI"""
        Z = pru.proj_alg_ric_newtonadi(mmat=self.M, fmat=self.F,
                                       jmat=self.J, bmat=self.bmat,
                                       wmat=self.W, z0=self.bmat,
                                       nwtn_adi_dict=
                                       self.nwtn_adi_dict)['zfac']

        # for '0' initial value --> z0 = None
        Z0 = pru.proj_alg_ric_newtonadi(mmat=self.M, fmat=self.F,
                                        jmat=self.J, bmat=self.bmat,
                                        wmat=self.W,
                                        nwtn_adi_dict=
                                        self.nwtn_adi_dict)['zfac']

        MtXM = self.M.T * np.dot(Z, Z.T) * self.M
        MtX0M = self.M.T * np.dot(Z0, Z0.T) * self.M

        self.assertTrue(np.allclose(MtXM, MtX0M))

        MtXb = self.M.T * np.dot(np.dot(Z, Z.T), self.bmat)

        FtXM = self.F.T * np.dot(Z, Z.T) * self.M
        PtW = np.dot(self.P.T, self.W)

        ProjRes = np.dot(self.P.T, np.dot(FtXM, self.P)) + \
            np.dot(np.dot(self.P.T, FtXM.T), self.P) -\
            np.dot(MtXb, MtXb.T) + \
            np.dot(PtW, PtW.T)

# TEST: result is 'projected' - riccati sol
        self.assertTrue(np.allclose(MtXM,
                                    np.dot(self.P.T, np.dot(MtXM, self.P))))

# TEST: check projected residual - riccati sol
        print np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)

        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)
                        < 1e-7)

    @unittest.skip('mvd to test_units_compfacres_compress ')
    def test_compress_algric_Z(self):
        Z = pru.proj_alg_ric_newtonadi(mmat=self.M, fmat=self.F,
                                       jmat=self.J, bmat=self.bmat,
                                       wmat=self.W, z0=self.bmat,
                                       nwtn_adi_dict=
                                       self.nwtn_adi_dict)['zfac']

        print '\ncompressing Z from {0} to {1} columns:'.\
            format(Z.shape[1], self.compn*self.W.shape[1])

        Zred = pru.compress_Z(Z, k=self.compn*self.W.shape[1])

        difn, zzn, zzrn = \
            lau.comp_sqfnrm_factrd_diff(Z, Zred, ret_sing_norms=True)

        print '\n || ZZ - ZZred||_F || / ||ZZ|| = {0}\n'.\
            format(np.sqrt(difn/zzn))

        vec = np.random.randn(Z.shape[0], 1)

        print '||(ZZ_red - ZZ )*testvec|| / ||ZZ*testvec|| = {0}'.\
            format(np.linalg.norm(np.dot(Z, np.dot(Z.T, vec)) -
                   np.dot(Zred, np.dot(Zred.T, vec))) /
                   np.linalg.norm(np.dot(Zred, np.dot(Zred.T, vec))))

        self.assertTrue(True)

suite = unittest.TestLoader().loadTestsFromTestCase(TestProjLyap)
unittest.TextTestRunner(verbosity=2).run(suite)
