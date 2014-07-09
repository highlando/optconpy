import unittest

import numpy as np
import sadptprj_riclyap_adi.proj_ric_utils as pru
import dolfin_navier_scipy.dolfin_to_sparrays as dtn
import scipy.sparse.linalg as spsla
import sadptprj_riclyap_adi.lin_alg_utils as lau
import scipy.sparse as sps

from dolfin_navier_scipy.problem_setups import drivcav_fems


class TestProjLyap(unittest.TestCase):

    def test_lyap_ress_compress(self):
        """comp of factrd lyap ress and compr of Z
        """
        N = 15
        NY = 5
        verbose = True
        k = None  # sing vals to keep
        thresh = 1e-6
        nwtn_adi_dict = dict(adi_max_steps=250,
                             adi_newZ_reltol=1e-8,
                             nwtn_max_steps=24,
                             nwtn_upd_reltol=4e-8,
                             nwtn_upd_abstol=4e-8,
                             verbose=verbose)

        femp = drivcav_fems(N)
        stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], nu=1)

        # remove the freedom in the pressure
        stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
        stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]

        # reduce the matrices by resolving the BCs
        (stokesmatsc,
         rhsd_stbc,
         invinds,
         bcinds,
         bcvals) = dtn.condense_sysmatsbybcs(stokesmats,
                                             femp['diribcs'])

        M = stokesmatsc['M']
        J = stokesmatsc['J']
        NV = M.shape[0]

        F = - stokesmatsc['M'] - 0.1*stokesmatsc['A'] - \
            sps.rand(NV, NV, density=0.03, format='csr')

        W = np.random.randn(NV, NY)

        nwtn_adi_dict = dict(adi_max_steps=50,
                             adi_newZ_reltol=1e-11,
                             nwtn_max_steps=24,
                             nwtn_upd_reltol=4e-7,
                             nwtn_upd_abstol=4e-7,
                             full_upd_norm_check=True,
                             verbose=verbose)

        Z = pru.solve_proj_lyap_stein(amat=F, mmat=M,
                                      jmat=J, wmat=W,
                                      adi_dict=nwtn_adi_dict)['zfac']

        MtZ = M.T * Z
        MtXM = np.dot(M.T*Z, Z.T*M)
        FtXM = F.T * np.dot(Z, Z.T) * M

        Mlu = spsla.factorized(M.tocsc())
        MinvJt = lau.app_luinv_to_spmat(Mlu, J.T)
        Sinv = np.linalg.inv(J * MinvJt)
        P = np.eye(NV) - np.dot(MinvJt, Sinv * J)

        PtW = np.dot(P.T, W)

        ProjRes = np.dot(P.T, np.dot(FtXM, P)) + \
            np.dot(np.dot(P.T, FtXM.T), P) + \
            np.dot(PtW, PtW.T)

        resn = np.linalg.norm(ProjRes)
        ownresn = np.sqrt(pru.comp_proj_lyap_res_norm(Z, F, M, W, J))

        # test smart fnorm comp
        self.assertTrue((np.allclose(np.linalg.norm(MtXM),
                        np.linalg.norm(np.dot(MtZ.T, MtZ)))))

        # test smart comp of ress
        self.assertTrue(np.allclose(resn, ownresn))

        # reduction of Z
        Zred = pru.compress_Zsvd(Z, k=k, thresh=thresh, shplot=True)
        MtZr = M.T * Zred
        MtXMr = np.dot(MtZr, MtZr.T)

        # TEST: reduction is 'projected'
        self.assertTrue((np.allclose(MtXMr, np.dot(P.T, np.dot(MtXMr, P)))))

        # TEST: diff in apprx
        self.assertTrue(np.allclose(np.linalg.norm(np.dot(MtZ.T, MtZ)),
                        np.linalg.norm(np.dot(MtZr.T, MtZr))))

        # print 'norm of red res: ', np.linalg.norm(ProjRes)
        ownresr = np.sqrt(pru.comp_proj_lyap_res_norm(Zred, F, M, W, J))

        self.assertTrue(np.allclose(ownresr, resn))
