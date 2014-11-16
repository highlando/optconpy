import plot_output as plo
import optcont_main as ocm
import matplotlib.pyplot as plt

path = '/home/heiland/work/papers/hei13_daericflow/pics/'

# extra options for tikz
extra = set(['ytick={-0.2, 0, 0.2}', 'xtick={0,0.2}'])

plt.figure(111)
fname = "pubpics/tds1_gamma0.001_alpha1e-9"
jsf = ocm.load_json_dicts(fname)
plo.plot_optcont_json(jsf, extra=extra, fname=path + 'tdgE-3aE-9')

plt.figure(112)
fname = "pubpics/sss1_gamma0.001_alpha1e-9"
jsf = ocm.load_json_dicts(fname)
plo.plot_optcont_json(jsf, extra=extra, fname=path + 'ssgE-3aE-9')
