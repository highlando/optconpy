import numpy as np
#from scipy.io import loadmat, savemat
from dolfin_to_nparrays import expand_vp_dolfunc


def output_paraview(tip, femp, vp=None, t=None):
    """write the paraview output for a solution vector vp

    """

    v, p = expand_vp_dolfunc(V=femp['V'], Q=femp['Q'], vp=vp,
                             invinds=femp['invinds'],
                             diribcs=femp['diribcs'])

    v.rename('v', 'velocity')
    p.rename('p', 'pressure')

    tip['vfile'] << v, t
    tip['pfile'] << p, t


def save_npa(v, fstring='not specified yet'):
    np.save(fstring, v)
    return


def load_npa(fstring):
    return np.load(fstring+'.npy')
