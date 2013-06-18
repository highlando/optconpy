import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import dolfin_to_nparrays as dtn
import proj_ric_utils as pru

from optcont_main import drivcav_fems


femp = drivcav_fems(20)

stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], 1e-3)

(stokesmatsc, 
        rhsd_stbc, 
        invinds, 
        bcinds, 
        bcvals) = dtn.condense_sysmatsbybcs(stokesmats, femp['diribcs'])
A, B, M = stokesmatsc['A'], stokesmatsc['B'], stokesmatsc['M'] 

pru.solve_proj_lyap_stein(At=-A, B=B, Mt=M, W=np.ones((len(invinds),2)))
# Minv = spsla.splu(M)
