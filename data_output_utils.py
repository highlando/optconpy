import numpy as np
from dolfin import File
#from scipy.io import loadmat, savemat
from dolfin_to_nparrays import expand_vp_dolfunc 


def output_paraview(femp, vp=None, t=None, fstring=''):
    """write the paraview output for a solution vector vp

    """

    v, p = dtn.expand_vp_dolfunc(femp, vp=vp)

    File(fstring+'_vel.pvd') << v, t
    File(fstring+'_p.pvd') << p, t
 

def save_curv(v, fstring='not specified yet'):
    np.save(fstring, v)
    return

def load_curv(fstring):
    return np.load(fstring+'.npy')
