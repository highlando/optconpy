import numpy as np
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import krypy.linsys


def stokes_steadystate(matdict=None, rhsdict=None, add_a=None):
    """solve for the steady state

    matdict ... dict of sysmatrices of stokes
    rhsdict ... dict of right hand sides
    add_a   ... matrix to be added to 'A' 
    """
    Np = len(rhsdict['fp'])

    if add_a is not None:
        SysM1 = sps.hstack([matdict['A']+add_a, 
            -matdict['JT']],format='csr')
    else:
        SysM1 = sps.hstack([matdict['A'], -matdict['JT']],format='csr')

    SysM2 = sps.hstack([matdict['J'],sps.csr_matrix((Np,Np))],format='csr')
    A = sps.vstack([SysM1,SysM2],format='csr')

    rhs = np.vstack([rhsdict['fv'], rhsdict['fp']])

    vp = np.atleast_2d(spsla.spsolve(A,rhs)).T

    return vp

def apply_massinv(M, rhsa, output=None):
    """ inverse of mass or any other spd matrix applied

    to a rhs array 
    TODO: check cases for CG, spsolve, 
    """
    if output=='sparse':
        return spsla.spsolve(M, rhsa)

    else:
        mlu = spsla.splu(M)
        try:
            mirhs = np.copy(rhsa.todense())
        except AttributeError:
            mirhs = np.copy(rhsa)

        for ccol in range(mirhs.shape[1]):
            mirhs[:,ccol] = mlu.solve(mirhs[:,ccol])

        return mirhs


def apply_invsqrt_fromleft(M, rhsa, output=None):
    """apply the sqrt of the inverse of a mass matrix or other spd 

    """
    Z = scipy.linalg.cholesky(M.todense())
    # R = Z.T*Z  <-> R^-1 = Z^-1*Z.-T
    if output=='sparse':
        return sps.csc_matrix(rhsa*np.linalg.inv(Z))
    else:
        return rhsa*np.linalg.inv(Z)

def get_Sinv_smw(Alu, U=None, V=None):
    """ compute (the small) inverse of I-V.T*Ainv*U
    """
    aiu = np.zeros(U.shape)
    for ccol in range(U.shape[1]):
        try:
            aiu[:,ccol] = Alu.solve(U[:,ccol])
        except AttributeError:
            aiu[:,ccol] = spsla.spsolve(Alu,U[:,ccol])

    return np.linalg.inv(np.eye(U.shape[1])-np.dot(V.T,aiu))


def app_smw_inv(Alu, U=None, V=None, rhsa=None, Sinv=None):
    """compute the sherman morrison woodbury inverse 

    of 
        A - np.dot(U,V.T)

    applied to (array)rhs. 
    """

    auvirhs = np.zeros(rhsa.shape)
    for rhscol in range(rhsa.shape[1]):
        if Sinv is None:
            Sinv = get_Sinv_smw(Alu,U,V)

        crhs = rhsa[:,rhscol]
        # the corrected rhs: (I + U*Sinv*VT*Ainv)*rhs
        crhs = crhs + np.dot(U, np.dot(Sinv, 
                                    np.dot(V.T, Alu.solve(crhs))))

        try:
            # if Alu comes with a solve routine, e.g. LU-factored - fine
            auvirhs[:,rhscol] = Alu.solve(crhs)
        except AttributeError:
            auvirhs[:,rhscol] = spsla.spsolve(Alu,crhs)

    return auvirhs

def app_schurc_inv(M, J, veca):
    """ apply the inverse of the Schurcomplement 

    for M is strictly positive definite
    """

    def _schurc(cveca):
        try:
            # if M comes with a solve routine
            return J*M.solve(J.T*cveca.flatten())
        except AttributeError:
            # return J*krypy.linsys.cg(M, J.T*cveca, tol=1e-18)['xk']
            return J*spsla.spsolve(M, J.T*cveca)

    S = spsla.LinearOperator( (J.shape[0],J.shape[0]), matvec=_schurc,
                               dtype=np.float32)

    auveca = np.zeros(veca.shape)
    for ccol in range(veca.shape[1]):
        auveca[:,ccol] = krypy.linsys.cg(S, veca[:,ccol], tol=1e-16)['xk']

    return auveca


