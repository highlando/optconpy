import matplotlib.pyplot as plt


def plot_optcont_json(jsonfile):
    outputplot(jsonfile['tmesh'], jsonfile['ycomp'],
               ystar=jsonfile['ystar'])


def outputplot(tmesh, ycomp, ystar=None):
    if ystar is not None:
        plt.plot(tmesh, ycomp, tmesh, ystar)
    else:
        plt.plot(tmesh, ycomp)

    plt.show(block=False)
