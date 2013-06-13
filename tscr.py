import numpy as np
from dolfin import *
from dolfin_to_nparrays import expand_vp_dolfunc

mesh = UnitSquare(64, 64)

V = VectorFunctionSpace(mesh, "CG", 1)
u = Expression(('x[1]','0'))

ufun = project(u, V, solver_type='lu')
ufunbc = project(u, V, solver_type='lu')

# definition of the boundary
def boundary(x, on_boundary): 

# definition of the boundary
def boundary(x, on_boundary): 
    return (x[1] > 1.0 - DOLFIN_EPS 
        or x[0] > 1.0 - DOLFIN_EPS 
        or x[1] < DOLFIN_EPS 
        or x[0] < DOLFIN_EPS)

# take the function values as dirichlet data
diridata = u 
bc = DirichletBC(V, diridata, boundary)

# apply boundary conditions
bc.apply(ufunbc.vector())
    
plot(ufun-ufunbc)

print np.linalg.norm(ufun.vector().array() - ufunbc.vector().array())

print np.allclose(ufun.vector().array(), ufunbc.vector().array())

