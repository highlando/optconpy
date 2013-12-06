import numpy as np
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import krypy.linsys


def app_prj_via_sadpnt(amat=None, jmat=None, rhsv=None,
                       jmatT=None, umat=None, vmat=None,
                       transposedprj=False):
    """apply projection via solving a sadpnt problem

    Pv = sadpointmat^-1 * amat * v

    P.T v = amat.T * sadpointmat^-T *  v
    
    """
    if jmatT is None:
        jmatT = jmat.T
    if jmat is None:
        jmat = jmatT.T


    if transposedprj:
        return amat.T * solve_sadpnt_smw(amat=amat.T, jmat=jmatT.T, 
                                         rhsv=rhsv, jmatT=jmat.T,
                                         )[:amat.shape[0], :]

    else:
        if umat is None and vmat is None:
            arhsv = amat * rhsv
        else:
            arhsv = amat * rhsv - \
                np.dot(umat, np.dot(vmat, rhsv))

        return solve_sadpnt_smw(amat=amat, jmat=jmat, rhsv=arhsv, 
                                jmatT=jmatT)[:amat.shape[0], :]


def solve_sadpnt_smw(amat=None, jmat=None, rhsv=None,
                     jmatT=None, umat=None, vmat=None,
                     rhsp=None):
    """solves with
            A - np.dot(U,V)    J.T  *  X   =   rhsv
            J                   0              rhsp
    """

    nnpp = jmat.shape[0]

    if jmatT is None:
        jmatT = jmat.T
    if jmat is None:
        jmat = jmatT.T

    if rhsp is None:
        rhsp = np.zeros((nnpp, rhsv.shape[1]))

    sysm1 = sps.hstack([amat, jmatT], format='csr')
    sysm2 = sps.hstack([jmat, sps.csr_matrix((nnpp, nnpp))], format='csr')
    mata = sps.vstack([sysm1, sysm2], format='csr')

    if sps.isspmatrix(rhsv):
        rhs = np.vstack([np.array(rhsv.todense()), rhsp])
    else:
        rhs = np.vstack([rhsv, rhsp])

    if umat is not None:
        vmate = sps.hstack([vmat, sps.csc_matrix((vmat.shape[0], nnpp))])
        umate = np.vstack([umat, np.zeros((nnpp, umat.shape[1]))])
    else:
        umate, vmate = None, None

    return app_smw_inv(mata, umat=umate, vmat=vmate, rhsa=rhs)


def stokes_steadystate(matdict=None, rhsdict=None, add_a=None):
    """solve for the steady state

    matdict ... dict of sysmatrices of stokes
    rhsdict ... dict of right hand sides
    add_a   ... matrix to be added to 'A'
    """
    Np = len(rhsdict['fp'])

    if add_a is not None:
        SysM1 = sps.hstack([matdict['A'] + add_a,
                            -matdict['JT']], format='csr')
    else:
        SysM1 = sps.hstack([matdict['A'], -matdict['JT']], format='csr')

    SysM2 = sps.hstack([matdict['J'], sps.csr_matrix((Np, Np))], format='csr')
    A = sps.vstack([SysM1, SysM2], format='csr')

    rhs = np.vstack([rhsdict['fv'], rhsdict['fp']])

    vp = np.atleast_2d(spsla.spsolve(A, rhs)).T

    return vp


def apply_massinv(M, rhsa, output=None):
    """ inverse of mass or any other spd matrix applied

    to a rhs array
    TODO: check cases for CG, spsolve,
    """
    if output == 'sparse':
        return spsla.spsolve(M, rhsa)

    else:
        mlusolve = spsla.factorized(M.tocsc())
        try:
            mirhs = np.copy(rhsa.todense())
        except AttributeError:
            mirhs = np.copy(rhsa)

        for ccol in range(mirhs.shape[1]):
            mirhs[:, ccol] = mlusolve(mirhs[:, ccol])

        return mirhs


def apply_invsqrt_fromleft(M, rhsa, output=None):
    """apply the sqrt of the inverse of a mass matrix or other spd

    TODO: cases for dense and sparse INPUTS
    """
    Z = scipy.linalg.cholesky(M.todense())
    # R = Z.T*Z  <-> R^-1 = Z^-1*Z.-T
    if output == 'sparse':
        return sps.csc_matrix(rhsa * np.linalg.inv(Z))
    else:
        return np.dot(rhsa, np.linalg.inv(Z))


def get_Sinv_smw(amat_lu, umat=None, vmat=None):
    """ compute (the small) inverse of I-V*Ainv*U
    """
    aiu = np.zeros(umat.shape)

    for ccol in range(umat.shape[1]):
        try:
            aiu[:, ccol] = amat_lu(umat[:, ccol])
        except TypeError:
            aiu[:, ccol] = spsla.spsolve(amat_lu, umat[:, ccol])

    if sps.isspmatrix(vmat):
        return np.linalg.inv(np.eye(umat.shape[1]) - vmat * aiu)
    else:
        return np.linalg.inv(np.eye(umat.shape[1]) - np.dot(vmat, aiu))


def app_luinv_to_spmat(alu_solve, Z):
    """ compute A.-1*Z  where A comes factored

    and with a solve routine"""

    Z.tocsc()
    ainvz = np.zeros(Z.shape)
    for ccol in range(Z.shape[1]):
        ainvz[:, ccol] = alu_solve(Z[:, ccol].toarray().flatten())

    return ainvz


def app_smw_inv(amat, umat=None, vmat=None, rhsa=None, Sinv=None,
                savefactoredby=5):
    """compute the sherman morrison woodbury inverse

    of
        A - np.dot(U,V)

    applied to (array)rhs.
    """
    

    if rhsa.shape[1] >= savefactoredby:
        try:
            alu = spsla.factorized(amat)
        except NotImplementedError:
            alu = amat
    else:
        alu = amat

    auvirhs = np.zeros(rhsa.shape)
    for rhscol in range(rhsa.shape[1]):
        crhs = rhsa[:, rhscol]
        # branch with u and v present
        if umat is not None:
            if Sinv is None:
                Sinv = get_Sinv_smw(alu, umat, vmat)

            # the corrected rhs: (I + U*Sinv*V*Ainv)*rhs
            try:
                # if Alu comes factorized, e.g. LU-factored - fine
                aicrhs = alu(crhs)
            except TypeError:
                aicrhs = spsla.spsolve(alu, crhs)

            if sps.isspmatrix(vmat):
                crhs = crhs + np.dot(umat, np.dot(Sinv, vmat * aicrhs))
            else:
                crhs = crhs + np.dot(umat, np.dot(Sinv, np.dot(vmat, aicrhs)))

        try:
            auvirhs[:, rhscol] = alu(crhs)
        except TypeError:
            auvirhs[:, rhscol] = spsla.spsolve(alu, crhs)

    return auvirhs


def app_schurc_inv(M, J, veca):
    """ apply the inverse of the Schurcomplement

    for M is strictly positive definite
    """

    def _schurc(cveca):
        try:
            # if M comes with a solve routine
            return J * M(J.T * cveca.flatten())
        except TypeError:
            return J * spsla.spsolve(M, J.T * cveca)

    S = spsla.LinearOperator((J.shape[0], J.shape[0]), matvec=_schurc,
                             dtype=np.float32)

    auveca = np.zeros(veca.shape)
    for ccol in range(veca.shape[1]):
        auveca[:, ccol] = krypy.linsys.cg(S, veca[:, ccol], tol=1e-16)['xk']

    return auveca


def comp_sqfnrm_factrd_diff(zone, ztwo, ret_sing_norms=False):
    """compute the squared Frobenius norm of z1*z1.T - z2*z2.T

    using the linearity traces and that tr.(z1.dot(z2)) = tr(z2.dot(z1))
    and that tr(z1.dot(z1.T)) is faster computed via (z1*z1.sum(-1)).sum()
    """

    ata = np.dot(zone.T, zone)
    btb = np.dot(ztwo.T, ztwo)
    atb = np.dot(zone.T, ztwo)

    if ret_sing_norms:
        norm_z1 = (ata * ata).sum(-1).sum()
        norm_z2 = (btb * btb).sum(-1).sum()
        return (norm_z1 - 2 * (atb * atb).sum(-1).sum() + norm_z2,
                norm_z1,
                norm_z2)

    return (ata * ata).sum(-1).sum() -  \
        2 * (atb * atb).sum(-1).sum() + \
        (btb * btb).sum(-1).sum()


def comp_sqfnrm_factrd_sum(zone, ztwo, ret_sing_norms=False):
    """compute the squared Frobenius norm of z1*z1.T + z2*z2.T

    using the linearity traces and that tr.(z1.dot(z2)) = tr(z2.dot(z1))
    and that tr(z1.dot(z1.T)) is faster computed via (z1*z1.sum(-1)).sum()
    """

    ata = np.dot(zone.T, zone)
    btb = np.dot(ztwo.T, ztwo)
    atb = np.dot(zone.T, ztwo)

    if ret_sing_norms:
        norm_z1 = (ata * ata).sum(-1).sum()
        norm_z2 = (btb * btb).sum(-1).sum()
        return (norm_z1 + 2 * (atb * atb).sum(-1).sum() + norm_z2,
                norm_z1,
                norm_z2)

    return (ata * ata).sum(-1).sum() +  \
        2 * (atb * atb).sum(-1).sum() + \
        (btb * btb).sum(-1).sum()


def comp_sqfnrm_factrd_lyap_res(A, B, C):
    """compute the squared Frobenius norm of A*B.T + B*A.T + C*C.T

    using the linearity traces and that tr.(z1.dot(z2)) = tr(z2.dot(z1))
    and that tr(z1.dot(z1.T)) is faster computed via (z1*z1.sum(-1)).sum()
    """

    ata = np.dot(A.T, A)
    atb = np.dot(A.T, B)
    atc = np.dot(A.T, C)
    btb = np.dot(B.T, B)
    btc = np.dot(B.T, C)
    ctc = np.dot(C.T, C)

    return 2 * (btb * ata).sum(-1).sum() +  \
        2 * (atb * atb.T).sum(-1).sum() + \
        4 * (btc.T * atc.T).sum(-1).sum() + \
        (ctc * ctc).sum(-1).sum()


def comp_uvz_spdns(umat, vmat, zmat):
    """comp u*v*z for sparse or dense u or v"""

    if sps.isspmatrix(vmat):
        vz = vmat * zmat
    else:
        vz = np.dot(vmat, zmat)
    if sps.isspmatrix(umat):
        return umat * vz
    else:
        return np.dot(umat, vz)
