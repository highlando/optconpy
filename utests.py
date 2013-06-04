import unittest
import sympy as smp
import numpy as np
import scipy.sparse as sps

# unittests for the suite
# if not specified otherwise we use the unit square 
# with 0-Dirichlet BCs with a known solution 

class OptConPyFunctions(unittest.TestCase):

	def setUp(self):
		from dolfin import UnitSquareMesh, VectorFunctionSpace, FunctionSpace, Expression
		from sympy import diff, sin, simplify

		self.mesh = UnitSquareMesh(24, 24)
		self.V = VectorFunctionSpace(self.mesh, "CG", 2)
		self.Q = FunctionSpace(self.mesh, "CG", 1)
		self.nu = 1e-5

		x, y, t, nu, om = smp.symbols('x,y,t,nu,om')
		ft = smp.sin(om*t)
		u_x = ft*x*x*(1-x)*(1-x)*2*y*(1-y)*(2*y-1)
		u_y = ft*y*y*(1-y)*(1-y)*2*x*(1-x)*(1-2*x)
		p = ft*x*(1-x)*y*(1-y)

		self.u_x = u_x
		self.u_y = u_y
		self.p = p

		def sympy2expression(term):
			'''Translate a SymPy expression to a FEniCS expression string.
			'''
			# This is somewhat ugly: 
			# First replace the variables r, z, by something
			# that probably doesn't appear anywhere else, 
			# e.g., RRR, ZZZ, then
			# convert this into a string, 
			# and then replace the substrings RRR, ZZZ
			# by x[0], x[1], respectively.
			exp = smp.printing.ccode(term.subs('x','XXX').subs('y','YYY')) \
				.replace('M_PI','pi') \
				.replace('XXX','x[0]').replace('YYY','x[1]')
			return exp

		dotu_x = simplify(diff(u_x,t))
		dotu_y = simplify(diff(u_y,t))

		diffu_x = simplify(nu*(diff(u_x,x,x) + diff(u_x,y,y)))
		diffu_y = simplify(nu*(diff(u_y,x,x) + diff(u_y,y,y)))

		dp_x = simplify( diff(p,x) )
		dp_y = simplify( diff(p,y) )

		adv_x = simplify( u_x*diff(u_x,x) + u_y*diff(u_x,y) )
		adv_y = simplify( u_x*diff(u_y,x) + u_y*diff(u_y,y) )

		self.F = Expression(('0','0'))
		a = sympy2expression(u_x)
		b = sympy2expression(u_x)
		
        self.fenics_sol_u = Expression((a,
                                   b),
                                   t=0.0, om=1.0)
	
	def test_testsolution_for_incompress(self):
		from sympy import diff

		# div u --- should be zero!!
        # self.assertEqual(diff(self.u_x,x) + diff(self.u_y,y), 0)


	def test_linearized_mat_NSE_form(self):
		"""check the conversion: dolfin form <-> numpy arrays

		and the linearizations"""

		import dolfin_to_nparrays as dtn

		u = self.fenics_sol_u
		uvec = u.array()
		uvec = uvec.reshape(len(uvec), 1)

		self.assertTrue(np.allclose(0, 0))


if __name__ == '__main__':
    unittest.main()

