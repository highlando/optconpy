import dolfin
import scipy.sparse.linalg as spsla
import cont_obs_utils as cou

dolfin.parameters.linear_algebra_backend = "uBLAS"


def check_outop(NV=20, NY=4, odcoo=dict(xmin=0.45,
                                        xmax=0.55,
                                        ymin=0.6,
                                        ymax=0.8)):
    """For some combinations of NV and NY
    the output operator C is badly sampled.
    This script checks how the vel = [1,1] is
    mapped to Cv"""

    mesh = dolfin.UnitSquareMesh(NV, NV)
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    dolfin.plot(mesh)

    exv = dolfin.Expression(('1', '2'))
    testv = dolfin.interpolate(exv, V)

    # check the C
    print ('assembling the output operator... ' +
           '(NV = {0}, NY = {1})').format(NV, NY)
    MyC, My = cou.get_mout_opa(odcoo=odcoo, V=V, NY=NY, NV=NV)
    print 'done!'
    testvi = testv.vector().array()
    testy = spsla.spsolve(My, MyC * testvi)

    # signal space
    ymesh = dolfin.IntervalMesh(NY - 1, odcoo['ymin'], odcoo['ymax'])
    Y = dolfin.FunctionSpace(ymesh, 'CG', 1)

    y1 = dolfin.Function(Y)
    y1.vector().set_local(testy[:NY])
    y1.rename("x-comp of C*v", "signal")
    dolfin.plot(y1)

    dolfin.interactive(True)

if __name__ == '__main__':
    check_outop()
