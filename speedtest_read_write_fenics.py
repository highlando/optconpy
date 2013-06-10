from timeit import timeit
setup = """
from dolfin import UnitSquareMesh, FunctionSpace, Expression, Function, File, project
from scipy.io import loadmat, savemat
import numpy as np
import os

mesh = UnitSquareMesh(128, 128)
V = FunctionSpace(mesh, "CG", 2)

u = Expression('x[0]*x[0]')
ufun = project(u, V)

File('saved_u.xml') << ufun
u_old = Function(V, 'saved_u.xml') 
uvec = ufun.vector().array().reshape(len(ufun.vector()),1)

savemat('saved_u', { 'uvec': uvec }, oned_as='column')
np.save('saved_u',uvec)
uvec_old = loadmat('saved_u')['uvec']
u_old = Function(V)
u_old.vector().set_local(uvec_old[:,0])

"""

print '10 times \n read and \n write xml-files \ntakes '
print timeit("File('saved_u.xml') << ufun, os.remove('saved_u.xml')", setup=setup, number=10)
print timeit("u_old = Function(V, 'saved_u.xml')", setup=setup, number=10)

# The scipy procedure
print '\n10 times \n convert to numpyarray \n save and \n load binaries via scipy io \n convert back to dolfin function \ntakes '
print timeit("uvec = ufun.vector().array().reshape(len(ufun.vector()),1)",
        setup=setup, number=10)
print timeit("savemat('saved_u', { 'uvec': uvec }, oned_as='column'), os.remove('saved_u.mat')",setup=setup, number=10)
print timeit("uvec_old = loadmat('saved_u')['uvec']",setup=setup, number=10)
print timeit("u_old.vector().set_local(uvec_old[:,0])",
        setup=setup, number=10)

# The numpy way
print '\n10 times \n convert to numpyarray \n save and \n load binaries \n convert back to dolfin function \ntakes '
print timeit("uvec = ufun.vector().array().reshape(len(ufun.vector()),1)",
        setup=setup, number=10)
print timeit("np.save('saved_u',uvec), os.remove('saved_u.npy')",setup=setup, number=10)
print timeit("uvec_old = np.load('saved_u.npy')",setup=setup, number=10)
print timeit("u_old.vector().set_local(uvec_old[:,0])",
        setup=setup, number=10)
