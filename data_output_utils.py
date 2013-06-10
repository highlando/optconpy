from dolfin import *
from scipy.io import loadmat, savemat
import numpy as np

def expand_vp_dolfunc(femp, vp=None, vc=None, pc=None, pdof=None):
    """expand v and p to the dolfin function representation
    
    pdof = pressure dof that was set zero
    """

    v = Function(femp['V'])
    p = Function(femp['Q'])

    invinds = femp['innerinds']

    if vp is not None:
        if vp.ndim == 1:
            vc = vp[:len(invinds)].reshape(len(invinds),1)
            pc = vp[len(invinds):].reshape(femp['Q'].dim()-1,1)
        else:
            vc = vp[:len(invinds),:]
            pc = vp[len(invinds):,:]

    ve = np.zeros((femp['V'].dim(),1))

    # fill in the boundary values
    for bc in femp['diribcs']:
        bcdict = bc.get_boundary_values()
        ve[bcdict.keys(),0] = bcdict.values()

    ve[invinds] = vc

    pe = np.vstack([pc,[0]])

    v.vector().set_local(ve)
    p.vector().set_local(pe)

    return v, p


def output_paraview(femp, vp=None, t=None, fstring=''):
    """write the paraview output for a solution vector vp

    """

    v, p = expand_vp_dolfunc(femp, vp=vp)

    File(fstring+'_vel.pvd') << v, t
    File(fstring+'_p.pvd') << p, t
 

def save_curv(v, fstring='not specified yet'):
    savemat(fstring, { 'v': v })

