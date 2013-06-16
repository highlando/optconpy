import numpy as np
import scipy.sparse as sps

import dolfin_to_nparrays as dtn
import proj_ric_utils as pru

from optcont_main import drivcav_fems


femp = drivcav_fems(10)

stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], tip['nu'])

(stokesmatsc, 
        rhsd_stbc, 
        invinds, 
        bcinds, 
        bcvals) = dtn.condense_sysmatsbybcs(stokesmats, femp['diribcs'])


