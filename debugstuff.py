import numpy as np
import dolfin_navier_scipy.data_output_utils as dou
import plot_output as plo
import optcont_main as ocm
import matplotlib.pyplot as plt


def plot_norms(tmesh, fbtdict):
    for t in tmesh:
        w = dou.load_npa(fbtdict[t]['w'])
        mtxtb = dou.load_npa(fbtdict[t]['mtxtb'])
        print t, np.linalg.norm(w), np.linalg.norm(mtxtb)


def plot_vel_norms(tmesh, veldict):
    print 'plotting vel norms'
    for t in tmesh:
        w = dou.load_npa(veldict[t])
        print t, np.linalg.norm(w)

if __name__ == '__main__':
    plt.figure(11)
    plt.title('stokes')
    dtd = "data/drivencavity__stokes__timeall_nu0.005_mesh20_Nts16.0__sigout"
    jsf = ocm.load_json_dicts(dtd)
    plo.plot_optcont_json(jsf)

    plt.figure(12)
    plt.title('steady state stokes')
    dtd = "data/stst_drivencavity__stokes__timeNone_nu0.005_mesh20_NtsNoneNV3042NY4NU4alphau1e-09gamma0.001__sigout"
    jsf = ocm.load_json_dicts(dtd)
    plo.plot_optcont_json(jsf)

    plt.figure(13)
    plt.title('oseen dc')
    dtd = "data/tdst_drivencavity__stokes__timeall_nu0.005_mesh20_NtsNoneNV3042NY4NU4alphau1e-09gamma0.001__sigout"
    jsf = ocm.load_json_dicts(dtd)
    plo.plot_optcont_json(jsf)

    plt.figure(14)
    plt.title('uncontrolled dc')
    dtd = "data/drivencavity__timeall_nu0.005_mesh20_Nts16.0__sigout"
    jsf = ocm.load_json_dicts(dtd)
    plo.plot_optcont_json(jsf)
# ## outsrc branch
# 0.0 0.402433121089 0.002477546117
# 0.0669872981078 0.395454141038 0.00253677135718
# 0.25 0.371843690385 0.00270857242965
# 0.5 0.322101589453 0.00297662454694
# 0.75 0.238319642114 0.00331968035717
# 0.933012701892 0.11845865888 0.00393562009696
# 1.0 0.0 0.00504927011523

# ## master branch
# 0.0 0.082608068971 0.0999058248638
# 0.0669872981078 0.0849975489378 0.102085394905
# 0.25 0.0915623193334 0.109855765402
# 0.5 0.100854612282 0.130162402951
# 0.75 0.105859525973 0.181597437395
# 0.933012701892 0.08060182361 0.305483045424
# 1.0 0.0 0.489182429102
