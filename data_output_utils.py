import numpy as np
import scipy.io
import json
from dolfin_to_nparrays import expand_vp_dolfunc
import matplotlib.pyplot as pl
from matplotlib2tikz import save as tikz_save


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


def save_npa(v, fstring='notspecified'):
    np.save(fstring, v)
    return


def load_npa(fstring):
    return np.load(fstring+'.npy')


def save_spa(sparray, fstring='notspecified'):
    scipy.io.mmwrite(fstring, sparray)


def load_spa(fstring):
    return scipy.io.mmread(fstring).tocsc()


def save_output_json(ycomp, tmesh, ystar=None, fstring=None):
    """save the signals to json for postprocessing"""
    if fstring is None:
        fstring = 'nonspecified_output'

    print 'dou.plot_js_outsigs("' + fstring + '")'
    jsfile = open(fstring, mode='w')
    jsfile.write(json.dumps(dict(ycomp=ycomp,
                                 tmesh=tmesh,
                                 ystar=ystar)))


def load_json_dicts(StrToJs):

    fjs = open(StrToJs)
    JsDict = json.load(fjs)
    return JsDict


def plot_js_outsigs(StrToJs):

    jsd = load_json_dicts(StrToJs)

    if jsd['ystar'] is not None:
        ysta = np.array(jsd['ystar'])
    else:
        ysta = None

    ycoa = np.array(jsd['ycomp'])
    tme = jsd['tmesh']
    print tme

    NY = ycoa.shape[1] / 2
    ymin = -0.2
    ymax = 0.2

    # myaxestyle = {'fontsize': 24,
    #               'weight': 'bold',
    #               'verticalalignment': 'top',
    #               'horizontalalignment': 'center',
    #               'ha': 'left'}

    def plot_save_fig(tme, ycoa, ysta=None, Title=None):
        pl.figure()

        if ysta is not None:
            pl.plot(tme, ysta, 'b', linewidth=2.0)

        for k in range(ycoa.shape[1]):
            pl.plot(tme, ycoa[:, k], color='r', linewidth=2.0)  # , tme, ysta)

        pl.ylim(ymin, ymax)

        tikz_save(Title + '.tex',
                  figureheight='\\figureheight',
                  figurewidth='\\figurewidth'
                  )
        return
    if ysta is not None:
        ystax = ysta[:, 0]
        ystay = ysta[:, -1]
    else:
        ystax, ystay = None, None

    plot_save_fig(tme, ycoa[:, :NY], ysta=ystax, Title=StrToJs+'xcomp')
    plot_save_fig(tme, ycoa[:, NY:], ysta=ystay, Title=StrToJs+'ycomp')

    pl.show()
