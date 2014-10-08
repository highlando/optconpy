import matplotlib.pyplot as plt


def plot_optcont_json(jsonfile, fname='notspecified'):
    outputplot(jsonfile['tmesh'], jsonfile['ycomp'],
               ystar=jsonfile['ystar'], fname=fname)


def outputplot(tmesh, ycomp, ystar=None, fname='notspecified'):
    from matplotlib2tikz import save as tikz_save

    if ystar is not None:
        plt.plot(tmesh, ystar, color='b', linewidth=2.0)

    lines = plt.plot(tmesh, ycomp)
    plt.setp(lines, color='r', linewidth=2.0)

    plt.yticks([0.2, 0, -0.2])
    # plt.xticks([0, 2])

    tikz_save(fname + '.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth'
              )
    plt.show(block=False)
