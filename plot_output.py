import matplotlib.pyplot as plt


def plot_optcont_json(jsonfile, fname='notspecified', extra=None):
    outputplot(jsonfile['tmesh'], jsonfile['ycomp'],
               ystar=jsonfile['ystar'], fname=fname, extra=extra)


def outputplot(tmesh, ycomp, ystar=None, extra=None, fname='notspecified'):
    from matplotlib2tikz import save as tikz_save

    if ystar is not None:
        plt.plot(tmesh, ystar, color='b', linewidth=2.0)

    lines = plt.plot(tmesh, ycomp)
    plt.setp(lines, color='r', linewidth=2.0)

    plt.yticks([0.2, 0, -0.2])
    # plt.xticks([0, 2])

    tikz_save(fname + '.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth',
              extra=extra
              )
    plt.show(block=False)
