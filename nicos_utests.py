import unittest
from dolfin import *
import pymelt.induction_heating.navier_stokes_cartesian as ns_car
import pymelt.induction_heating.navier_stokes_cylindrical as ns_cyl
import sympy as smp
import numpy as np
# ==============================================================================
class TestNavierStokesCartesian(unittest.TestCase):
    # TODO Add test case that checks nontrivial analytic solution.
    # --------------------------------------------------------------------------
    def setUp(self):
        return
    # --------------------------------------------------------------------------
    def _sympy2expression(self, term):
        '''Translate a SymPy expression to a FEniCS expression string.
        '''
        # This is somewhat ugly: First replace the variables r, z, by something
        # that probably doesn't appear anywhere else, e.g., RRR, ZZZ, then
        # convert this into a string, and then replace the substrings RRR, ZZZ
        # by x[0], x[1], respectively.
        exp = smp.printing.ccode(term.subs('x','XXX').subs('y','YYY')) \
            .replace('M_PI','pi') \
            .replace('XXX','x[0]').replace('YYY','x[1]')

        return exp
    # --------------------------------------------------------------------------
    def _test_chorin(self, mesh, u_x, u_y, p, T, dt,
                     mu_default = 1.0,
                     rho_default = 1.0
                     ):
        '''Test against exact solution.
        '''

        x, y, t, mu, rho = smp.symbols('x,y,t,mu,rho')

        # Make sure that the exact solution is indeed analytically div-free.
        d = smp.diff(u_x, x) + smp.diff(u_y, y)
        d = smp.simplify(d)
        self.assertEqual(d, 0)

        # Translate solution into a FEniCS expression.
        fenics_sol_u = Expression((self._sympy2expression(u_x),
                                   self._sympy2expression(u_y)),
                                   t=0.0)
        fenics_sol_p = Expression(self._sympy2expression(p), t=0.0)


        # Get right-hand side associated with this solution, i.e., according
        # the Navier-Stokes
        #
        #     rho (du_x/dt + u_x du_x/dx + u_y du_x/dy) = - dp/dx + mu [d^2u_x/dx^2 + d^2u_x/dy^2] + f_x,
        #     rho (du_y/dt + u_x du_y/dx + u_y du_y/dy) = - dp/dx + mu [d^2u_y/dx^2 + d^2u_y/dy^2] + f_y,
        #     du_x/dx + du_y/dy = 0.
        #
        #     rho (du/dt + (u.\nabla)u) = -\nabla p + mu [\div(\nabla u)] + f,
        #     div(u) = 0.
        #
        f_x = rho * (smp.diff(u_x, t)
                    + u_x * smp.diff(u_x, x)
                    + u_y * smp.diff(u_x, y)) \
            + smp.diff(p, x) \
            - mu * (smp.diff(u_x,x,x) + smp.diff(u_x, y, y))
        f_x = smp.simplify(f_x)
        f_y = rho * (smp.diff(u_y, t)
                    + u_x * smp.diff(u_y, x)
                    + u_y * smp.diff(u_y, y)) \
            + smp.diff(p, y) \
            - mu * (smp.diff(u_y, x, x) + smp.diff(u_y, y, y))
        f_y = smp.simplify(f_y)

        # Same for f.
        f_x0 = self._sympy2expression(f_x)
        f_y0 = self._sympy2expression(f_y)
        #print f_x0
        #print
        #print f_y0
        fenics_rhs = Expression((f_x0, f_y0),
                                mu=mu_default, rho=rho_default, t=0.0)
        V = FunctionSpace(mesh, 'CG', 2)

        #plot(project(fenics_rhs, V*V))
        #interactive()
        #exit()

        #for t in np.linspace(0.0, 1.0, 20):
        #    fenics_rhs.t = t
        #    plot(project(fenics_rhs, V*V))
        #    interactive()

        #for t in np.linspace(0.0, 1.0, 20):
        #    fenics_sol_u.t = t
        #    plot(project(fenics_sol_u, V*V))
        #    interactive()
        #exit()

        fenics_sol_u.t = 0.0
        u = Function(V*V)
        u_1 = project(fenics_sol_u, V*V)
        # Perform one Navier-Stokes step.
        t = 0.0
        while t < T + DOLFIN_EPS:
            # Set time.
            fenics_rhs.t = t
            fenics_sol_u.t = t
            #u, p = ns_car.chorin_step(mesh, u_1, dt,
            u, p = ns_car.chorin_step_anders(mesh, u_1, dt,
                                      mu_default, rho_default,
                                      fenics_rhs,
                                      velocity_boundary_conditions = fenics_sol_u,
                                      verbose = False,
                                      compute_residuals = True
                                      )
            t += dt
            #plot(u)
            #interactive()
            u_1.assign(u)

        return u, p
    # --------------------------------------------------------------------------
    #def test_flat(self):
    #    '''Test that no velocity is generated in a box without in- or outflow.
    #    '''
    #    mesh = UnitSquareMesh(20, 20)
    #    V = FunctionSpace(mesh, 'CG', 2)
    #    u_1 = project(Constant((0.0,0.0)), V*V)
    #    dt = 1.0e-1
    #    mu = 1.0
    #    rho = 1.0
    #    g = Constant((0.0, -9.81))
    #    # Perform one Navier-Stokes step.
    #    u, p = ns_car.chorin_step(mesh, u_1, dt,
    #                              mu, rho,
    #                              rho*g,
    #                              verbose = False,
    #                              compute_residuals = False
    #                              )
    #    self.assertAlmostEqual(norm(u), 0.0, delta=1.0e-13)

    #    return
    ## --------------------------------------------------------------------------
    #def test_curl(self):
    #    '''Test against exact solution curl.
    #    '''
    #    # Set domain/mesh: rectangle [1,2]x[-0.5,0.5].
    #    xlims = [1, 2]
    #    ylims = [-0.5, 0.5]

    #    mesh = RectangleMesh(xlims[0], ylims[0], xlims[1], ylims[1],
    #                         16, 16, 'crossed')

    #    x, y, t, mu, rho = smp.symbols('x,y,t,mu,rho')
    #    # Choose the solution something that cannot exactly be expressed by
    #    # polynomials.
    #    # Note that the exact solution is indeed div-free.
    #    xmid = 0.5 * (xlims[0] + xlims[1])
    #    ymid = 0.5 * (ylims[0] + ylims[1])
    #    u_x = -smp.sin(smp.pi*t) * smp.sin(smp.pi*(y-ymid))
    #    u_y =  smp.sin(smp.pi*t) * smp.sin(smp.pi*(x-xmid))
    #    #p = 0.0
    #    p = -y

    #    dt = 4.0e-2
    #    T = dt - DOLFIN_EPS
    #    u, p = self._test_chorin(mesh, u_x, u_y, p, T, dt)

    #    #fenics_sol_u.t = T
    #    #self.assertAlmostEqual(errornorm(fenics_sol_u, u), 0.0, delta=1.0e-13)

    #    return
    # --------------------------------------------------------------------------
    def test_pistol_pete(self):
        '''Pistol Pete's example from Teodora I. Mitkova's text
        "Finite-Elemente-Methoden fur die Stokes-Gleichungen".
        '''

        x, y, t, mu, rho = smp.symbols('x,y,t,mu,rho')
        # Choose the solution something that cannot exactly be expressed by
        # polynomials.
        # Note that the exact solution is indeed div-free.
        u_x = x**2 * (1-x)**2 *2*y * (1-y) * (2*y-1)
        u_y = y**2 * (1-y)**2 *2*x * (1-x) * (1-2*x)
        p = x * (1-x) * y * (1-y)

        mesh = UnitSquareMesh(20, 20, 'crossed')

        dt = 1.0e-5
        T = dt - DOLFIN_EPS
        u_approx, p_approx = self._test_chorin(mesh, u_x, u_y, p, T, dt)

        fenics_sol_u = Expression((self._sympy2expression(u_x),
                                   self._sympy2expression(u_y)),
                                   t=0.0)
        # Check the solution.
        fenics_sol_u.t = T
        #fenics_sol_p.t = T
        #plot((fenics_sol_u - u)/norm(u))
        #plot((fenics_sol_p - p)/norm(p))
        ##plot(grad(project(fenics_sol_p - p,V)))
        #interactive()
        #exit()
        error_u = errornorm(fenics_sol_u, u_approx)/norm(fenics_sol_u, mesh=mesh)
        print('%e' % error_u)
        # p is only determined up to a constant, so best normalize both.
        #normalize(fenics_sol_p.vector())
        #normalize(p.vector())
        #print('%e' % (errornorm(fenics_sol_p, p)/norm(p)))
        #self.assertAlmostEqual(errornorm(fenics_sol_u, u)/norm(u), 0.0,
        #                       delta=1.0e-13)
        #self.assertAlmostEqual(errornorm(fenics_sol_p, p)/norm(p), 0.0,
        #                       delta=1.0e-13)

        #fenics_sol_u.t = T
        #self.assertAlmostEqual(errornorm(fenics_sol_u, u), 0.0, delta=1.0e-13)

        return
    # --------------------------------------------------------------------------
    def test_whirl(self):
        '''Whirl solution with no homogeneous boundary conditions.
        '''

        x, y, t, mu, rho = smp.symbols('x,y,t,mu,rho')
        # Choose the solution something that cannot exactly be expressed by
        # polynomials.
        # Note that the exact solution is indeed div-free.
        u_x = x**2 * (1-x)**2 *2*y * (1-y) * (2*y-1)
        u_y = y**2 * (1-y)**2 *2*x * (1-x) * (1-2*x)
        p = x * (1-x) * y * (1-y)

        mesh = UnitSquareMesh(20, 20, 'crossed')

        dt = 1.0e-5
        T = dt - DOLFIN_EPS
        u_approx, p_approx = self._test_chorin(mesh, u_x, u_y, p, T, dt)

        fenics_sol_u = Expression((self._sympy2expression(u_x),
                                   self._sympy2expression(u_y)),
                                   t=0.0)
        # Check the solution.
        fenics_sol_u.t = T
        #fenics_sol_p.t = T
        #plot((fenics_sol_u - u)/norm(u))
        #plot((fenics_sol_p - p)/norm(p))
        ##plot(grad(project(fenics_sol_p - p,V)))
        #interactive()
        #exit()
        error_u = errornorm(fenics_sol_u, u_approx)/norm(fenics_sol_u, mesh=mesh)
        print('%e' % error_u)
        # p is only determined up to a constant, so best normalize both.
        #normalize(fenics_sol_p.vector())
        #normalize(p.vector())
        #print('%e' % (errornorm(fenics_sol_p, p)/norm(p)))
        #self.assertAlmostEqual(errornorm(fenics_sol_u, u)/norm(u), 0.0,
        #                       delta=1.0e-13)
        #self.assertAlmostEqual(errornorm(fenics_sol_p, p)/norm(p), 0.0,
        #                       delta=1.0e-13)

        #fenics_sol_u.t = T
        #self.assertAlmostEqual(errornorm(fenics_sol_u, u), 0.0, delta=1.0e-13)

        return
    # --------------------------------------------------------------------------
    #def test_chorin_div(self):
    #    '''Test the divergence.
    #    '''
    #    mesh = UnitSquareMesh(20, 20)
    #    V = FunctionSpace(mesh, 'CG', 2)
    #    u_1 = project(Constant((0.0,0.0)), V*V)
    #    dt = 1.0e-1
    #    mu = 1.0
    #    rho = 1.0
    #    g = Expression(('cos(2*pi*x[0]*x[1])', 'sin(2*pi*x[0]*x[1])'))
    #    # Perform one Navier-Stokes step.
    #    u, p = cns.chorin_step(mesh, u_1, dt,
    #                           mu, rho,
    #                           rho*g,
    #                           verbose = True,
    #                           compute_residuals = True
    #                           )
    #    # Compute div(u).
    #    Q = FunctionSpace(mesh, 'CG', 1)
    #    d = Function(Q)
    #    x0 = Expression('x[0]')
    #    X0 = project(x0, Q)
    #    p1 = TrialFunction(Q)
    #    q = TestFunction(Q)
    #    a = p1*q*dx
    #    L = div(X0*u) * q * dx
    #    solve(a == L, d)
    #    self.assertAlmostEqual(norm(d), 0.0, delta=1.0e-13)

    #    return
    # --------------------------------------------------------------------------
## ==============================================================================
#class TestNavierStokesCylindrical(unittest.TestCase):
#    # TODO Add test case that checks nontrivial analytic solution.
#    # --------------------------------------------------------------------------
#    def setUp(self):
#        return
#    # --------------------------------------------------------------------------
#    #def test_chorin(self):
#    #    '''Test that no velocity is generated in a box without in- or outflow.
#    #    '''
#    #    mesh = UnitSquareMesh(20, 20)
#    #    V = FunctionSpace(mesh, 'CG', 2)
#    #    u_1 = project(Constant((0.0,0.0)), V*V)
#    #    dt = 1.0e-1
#    #    mu = 1.0
#    #    rho = 1.0
#    #    g = Constant((0.0, -9.81))
#    #    # Perform one Navier-Stokes step.
#    #    u, p = cns.chorin_step(mesh, u_1, dt,
#    #                           mu, rho,
#    #                           rho*g,
#    #                           verbose = False,
#    #                           compute_residuals = False
#    #                           )
#    #    self.assertAlmostEqual(norm(u), 0.0, delta=1.0e-13)
#
#    #    return
#    # --------------------------------------------------------------------------
#    def test_chorin2(self):
#        '''Test against exact solution.
#        '''
#        # Set domain/mesh: rectangle [1,2]x[-0.5,0.5].
#        rlims = [1,2]
#        zlims = [-0.5, 0.5]
#
#        r, z, t, mu, rho = smp.symbols('r,z,t,mu,rho')
#        # Choose the solution something that cannot exactly be expressed by
#        # polynomials.
#        rmid = 0.5 * (rlims[0] + rlims[1])
#        zmid = 0.5 * (zlims[0] + zlims[1])
#        u_r = -smp.sin(smp.pi*t) * smp.sin(smp.pi*(z-zmid))
#        u_z =  smp.sin(smp.pi*t) * smp.sin(smp.pi*(r-rmid))
#        p = -z
#        # Get right-hand side associated with this solution, i.e., according
#        # the Navier-Stokes
#        #
#        #     rho (du_r/dt + u_r du_r/dr + u_z du_r/dz) = - dp/dr + mu [1/r d/dr(r du_r/dr) + d^2u_r/dz^2 - u_r/r^2] + f_r,
#        #     rho (du_z/dt + u_r du_z/dr + u_z du_z/dz) = - dp/dz + mu [1/r d/dr(r du_z/dr) + d^2u_z/dz^2] + f_z,
#        #     1/r d(r u_r)/dr + du_z/dz = 0.
#        #
#        #     rho (du/dt + (u.\nabla)u) = -\nabla p + mu [1/r \div(r \nabla u) - e_r*u_r/r^2] + rho g,
#        #     1/r div(r u) = 0.
#        #
#        f_r = rho * (smp.diff(u_r, t)
#                    + u_r * smp.diff(u_r, r)
#                    + u_z * smp.diff(u_r, z)) \
#            + smp.diff(p, r) \
#            - mu * (1/r * smp.diff(r*smp.diff(u_r,r),r)
#                   + smp.diff(u_r, z, z)
#                   - u_r/(r*r))
#        f_r = smp.simplify(f_r)
#        f_z = rho * (smp.diff(u_z, t)
#                    + u_r * smp.diff(u_z, r)
#                    + u_z * smp.diff(u_z, z)) \
#            + smp.diff(p, z) \
#            - mu * (1/r * smp.diff(r*smp.diff(u_z,r),r)
#                   + smp.diff(u_z, z, z))
#        f_z = smp.simplify(f_z)
#
#        # Translate solution into a FEniCS expression.
#        # This is somewhat ugly: First replace the variables r, z, by something
#        # that probably doesn't appear anywhere else, e.g., RRR, ZZZ, then
#        # convert this into a string, and then replace the substrings RRR, ZZZ
#        # by x[0], x[1], respectively.
#        u_r0 = smp.printing.ccode(u_r.subs('r','RRR').subs('z','ZZZ')) \
#             .replace('M_PI','pi') \
#             .replace('RRR','x[0]').replace('ZZZ','x[1]')
#        # Same drill for u_z0.
#        u_z0 = smp.printing.ccode(u_z.subs('r','RRR').subs('z','ZZZ')) \
#             .replace('M_PI','pi') \
#             .replace('RRR','x[0]').replace('ZZZ','x[1]')
#        # Stich it all together.
#        fenics_sol_u = Expression((u_r0, u_z0), t=0.0)
#
#        # Same for f.
#        f_r0 = smp.printing.ccode(f_r.subs('r','RRR').subs('z','ZZZ')) \
#             .replace('M_PI','pi') \
#             .replace('RRR','x[0]').replace('ZZZ','x[1]')
#        f_z0 = smp.printing.ccode(f_z.subs('r','RRR').subs('z','ZZZ')) \
#             .replace('M_PI','pi') \
#             .replace('RRR','x[0]').replace('ZZZ','x[1]')
#        #print f_r0
#        #print
#        #print f_z0
#        mu_default = 1.0
#        rho_default = 1.0
#        fenics_rhs = Expression((f_r0, f_z0),
#                                mu=mu_default, rho=rho_default, t=0.0)
#
#        #for t in np.linspace(0.0, 1.0, 20):
#        #    fenics_rhs.t = t
#        #    plot(project(fenics_rhs, V*V))
#        #    interactive()
#
#        #for t in np.linspace(0.0, 1.0, 20):
#        #    fenics_sol_u.t = t
#        #    plot(project(fenics_sol_u, V*V))
#        #    interactive()
#        #exit()
#
#        mesh = RectangleMesh(rlims[0], zlims[0], rlims[1], zlims[1],
#                             20, 20, 'crossed')
#        V = FunctionSpace(mesh, 'CG', 2)
#
#        #plot(project(fenics_rhs, V*V))
#        #interactive()
#        #exit()
#
#        fenics_sol_u.t = 0.0
#        u = Function(V*V)
#        u_1 = project(fenics_sol_u, V*V)
#        dt = 1.0e-1
#        T = 1.0
#        # Perform one Navier-Stokes step.
#        t = 0.0
#        while t < T + DOLFIN_EPS:
#            # Set time.
#            fenics_rhs.t = t
#            fenics_sol_u.t = t
#            fenics_rhs = Expression(('0.0','-1.0'))
#            fenics_sol_u = Expression((0.0,0.0))
#            u, p = cns.chorin_step(mesh, u_1, dt,
#                                   mu_default, rho_default,
#                                   fenics_rhs,
#                                   velocity_boundary_conditions = fenics_sol_u,
#                                   verbose = False,
#                                   compute_residuals = False
#                                   )
#            t += dt
#            plot(u)
#            interactive()
#            u_1.assign(u)
#
#        exit()
#        fenics_sol_u.t = T
#        self.assertAlmostEqual(errornorm(fenics_sol_u, u), 0.0, delta=1.0e-13)
#
#        return
#    # --------------------------------------------------------------------------
#    #def test_chorin_div(self):
#    #    '''Test the divergence.
#    #    '''
#    #    mesh = UnitSquareMesh(20, 20)
#    #    V = FunctionSpace(mesh, 'CG', 2)
#    #    u_1 = project(Constant((0.0,0.0)), V*V)
#    #    dt = 1.0e-1
#    #    mu = 1.0
#    #    rho = 1.0
#    #    g = Expression(('cos(2*pi*x[0]*x[1])', 'sin(2*pi*x[0]*x[1])'))
#    #    # Perform one Navier-Stokes step.
#    #    u, p = cns.chorin_step(mesh, u_1, dt,
#    #                           mu, rho,
#    #                           rho*g,
#    #                           verbose = True,
#    #                           compute_residuals = True
#    #                           )
#    #    # Compute div(u).
#    #    Q = FunctionSpace(mesh, 'CG', 1)
#    #    d = Function(Q)
#    #    x0 = Expression('x[0]')
#    #    X0 = project(x0, Q)
#    #    p1 = TrialFunction(Q)
#    #    q = TestFunction(Q)
#    #    a = p1*q*dx
#    #    L = div(X0*u) * q * dx
#    #    solve(a == L, d)
#    #    self.assertAlmostEqual(norm(d), 0.0, delta=1.0e-13)
#
#    #    return
#    # --------------------------------------------------------------------------
# ==============================================================================
if __name__ == '__main__':
    unittest.main()
# ==============================================================================
