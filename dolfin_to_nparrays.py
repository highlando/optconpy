from dolfin import *
import numpy as np
import scipy.sparse as sps

parameters.linear_algebra_backend = "uBLAS"

def get_stokessysmats( V, Q, nu): # , velbcs ):
    """ Assembles the system matrices for Stokes equation

    in mixed FEM formulation, namely
        
        [ A  J' ] as [ Aa   Grada ] : W -> W'
        [ J  0  ]    [ Diva   0   ]
        
    for a given trial and test space W = V * Q and boundary conds.
    
    Plus the velocity mass matrix M.
    """

    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    ma = inner(u,v)*dx
    mp = inner(p,q)*dx
    aa = nu*inner(grad(u), grad(v))*dx 
    grada = div(v)*p*dx
    diva = q*div(u)*dx

    # Assemble system
    M = assemble(ma)
    A = assemble(aa)
    Grad = assemble(grada)
    Div = assemble(diva)
    MP = assemble(mp)

    # Convert DOLFIN representation to numpy arrays
    rows, cols, values = M.data()
    Ma = sps.csr_matrix((values, cols, rows))

    rows, cols, values = MP.data()
    MPa = sps.csr_matrix((values, cols, rows))

    rows, cols, values = A.data()
    Aa = sps.csr_matrix((values, cols, rows))

    rows, cols, values = Grad.data()
    JTa = sps.csr_matrix((values, cols, rows))

    rows, cols, values = Div.data()
    Ja = sps.csr_matrix((values, cols, rows))

    stokesmats = {'M':Ma,
            'A':Aa,
            'JT':JTa,
            'J':Ja,
            'MP':MPa}

    return stokesmats
    

def get_convmats(u0_dolfun=None, u0_vec=None, V=None, invinds=None,
                diribcs=None):
    """returns the matrices related to the linearized convection

    N1 ~ (u_0 \nabla u) v
    N2 ~ (u \nabla u_0) v

    where u_0 is the linearization point"""

    if u0_vec is not None:
        u0, p = expand_vp_dolfunc(vc=u0_vec, V=V, diribcs=diribcs,
                                    invinds=invinds)
    else:
        u0 = u0_dolfun

    u = TrialFunction(V)
    v = TestFunction(V)

    # Assemble system
    n1 = inner(grad(u)*u0, v)*dx
    n2 = inner(grad(u0)*u, v)*dx
    f3 = inner(grad(u0)*u0, v)*dx

    n1 = assemble(n1)
    n2 = assemble(n2)
    f3 = assemble(f3)

    # Convert DOLFIN representation to numpy arrays
    rows, cols, values = n1.data()
    N1 = sps.csr_matrix((values, cols, rows))

    rows, cols, values = n2.data()
    N2 = sps.csr_matrix((values, cols, rows))

    fv = f3.array()
    fv = fv.reshape(len(fv), 1)

    return N1, N2, fv

def setget_rhs(V, Q, fv, fp, t=None):

    if t is not None:
        fv.t = t
        fp.t = t
    elif hasattr(fv,'t') or hasattr(fp,'t'):
        Warning('No value for t specified') 

    v = TestFunction(V)
    q = TestFunction(Q)

    fv = inner(fv,v)*dx 
    fp = inner(fp,q)*dx

    fv = assemble(fv)
    fp = assemble(fp)

    fv = fv.array()
    fv = fv.reshape(len(fv), 1)

    fp = fp.array()
    fp = fp.reshape(len(fp), 1)

    rhsvecs = {'fv':fv,
            'fp':fp}

    return rhsvecs

def get_curfv(V, fv, invinds, tcur):
    """get the fv at innernotes at t=tcur

    """

    v = TestFunction(V)

    fv.t = tcur

    fv = inner(fv,v)*dx 

    fv = assemble(fv)

    fv = fv.array()
    fv = fv.reshape(len(fv), 1)

    return fv[invinds,:]


def get_convvec(u0_dolfun=None, V=None, u0_vec=None, femp=None):
    """return the convection vector e.g. for explicit schemes

    given a dolfin function or the coefficient vector
    """

    if u0_vec is not None:
        u0, p = expand_vp_dolfunc(vc=u0_vec, V=V, diribcs=diribcs,
                                    invinds=invinds)
    else:
        u0 = u0_dolfun

    v = TestFunction(V)
    ConvForm = inner(grad(u0)*u0, v)*dx

    ConvForm = assemble(ConvForm)
    ConvVec = ConvForm.array()
    ConvVec = ConvVec.reshape(len(ConvVec), 1)

    return ConvVec


def condense_sysmatsbybcs(stms, velbcs):
    """resolve the Dirichlet BCs and condense the sysmats

    to the inner nodes
    stms ... dictionary of the stokes matrices"""

    nv = stms['A'].shape[0]

    auxu = np.zeros((nv,1))
    bcinds = []
    for bc in velbcs:
        bcdict = bc.get_boundary_values()
        auxu[bcdict.keys(),0] = bcdict.values()
        bcinds.extend(bcdict.keys())

    # putting the bcs into the right hand sides
    fvbc = - stms['A']*auxu    # '*' is np.dot for csr matrices
    fpbc = - stms['J']*auxu
    
    # indices of the innernodes
    invinds = np.setdiff1d(range(nv),bcinds).astype(np.int32)

    # extract the inner nodes equation coefficients
    Mc = stms['M'][invinds,:][:,invinds]
    Ac = stms['A'][invinds,:][:,invinds]
    fvbc = fvbc[invinds,:]
    Jc  = stms['J'][:,invinds]
    JTc = stms['JT'][invinds,:]

    bcvals = auxu[bcinds]

    stokesmatsc = {'M':Mc,
            'A':Ac,
            'JT':JTc,
            'J':Jc}

    rhsvecsbc = {'fv':fvbc,
            'fp':fpbc}

    return stokesmatsc, rhsvecsbc, invinds, bcinds, bcvals


def condense_velmatsbybcs(A, velbcs):
    """resolve the Dirichlet BCs and condense the velmats

    to the inner nodes"""

    nv = A.shape[0]

    auxu = np.zeros((nv,1))
    bcinds = []
    for bc in velbcs:
        bcdict = bc.get_boundary_values()
        auxu[bcdict.keys(),0] = bcdict.values()
        bcinds.extend(bcdict.keys())

    # putting the bcs into the right hand sides
    fvbc = - A*auxu    # '*' is np.dot for csr matrices
    
    # indices of the innernodes
    invinds = np.setdiff1d(range(nv),bcinds).astype(np.int32)

    # extract the inner nodes equation coefficients
    Ac = A[invinds,:][:,invinds]
    fvbc = fvbc[invinds,:]

    return Ac, fvbc


def expand_vp_dolfunc(V=None, Q=None, invinds=None, diribcs=None, vp=None,
        vc=None, pc=None):
    """expand v [and p] to the dolfin function representation
    
    pdof = pressure dof that was set zero

    This function returns v as a dolfunc and - if specified - p
    Error if:
        - not enough input to expand v and maybe p
        - only p is to be expanded
    """

    if vp is not None:
        vc = vp[:len(invinds),:]
        pc = vp[len(invinds):,:]
        p = Function(Q)
    elif pc is not None:
        p = Function(Q)

    v = Function(V)
    ve = np.zeros((V.dim(),1))

    # fill in the boundary values
    for bc in diribcs:
        bcdict = bc.get_boundary_values()
        ve[bcdict.keys(),0] = bcdict.values()

    ve[invinds] = vc

    if pc is not None:
        pe = np.vstack([pc,[0]])
        p.vector().set_local(pe)
    else:
        p = None

    v.vector().set_local(ve)

    return v, p
