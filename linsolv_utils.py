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


