import numpy as np
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

