import sympy as smp
from sympy import diff, sin, cos, pi


x, y, t, nu, om = smp.symbols('x,y,t,nu,om')

ft = smp.sin(om*t)
u1 = ft*x*x*(1-x)*(1-x)*2*y*(1-y)*(2*y-1)
u2 = ft*y*y*(1-y)*(1-y)*2*x*(1-x)*(1-2*x)
p = ft*x*(1-x)*y*(1-y)

# Stokes case
rhs1 = smp.simplify(diff(u1,t) - nu*(diff(u1,x,x) + diff(u1,y,y)) + diff(p,x))
rhs2 = smp.simplify(diff(u2,t) - nu*(diff(u2,x,x) + diff(u2,y,y)) + diff(p,y))

#rhs3 = div u --- should be zero!!
rhs3 = diff(u1,x) + diff(u2,y)

# Advection (u.D)u
ad1 = smp.simplify( u1*diff(u1,x) + u2*diff(u1,y) )
ad2 = smp.simplify( u1*diff(u2,x) + u2*diff(u2,y) )

#rhs1 = smp.simplify(smp.simplify(rhs1) - smp.simplify(ad1))
#rhs2 = smp.simplify(rhs2 - ad2)
rhs3 = smp.simplify(rhs3)

print'rhs1 = \n\t%r \t\t\n' % smp.simplify( rhs1 + ad1 )
print'rhs2 = \n\t%r \t\t\n' % smp.simplify( rhs2 + ad2 )
print'rhs3 = \n\t%r \t\t\n' % rhs3

## regexp for replace ** by pow: %s/\(([xy0-9+-\* ]\+)\|x\|y\|\d\+\)\*\*\(\d\+\)/pow(\1,\2)/gc

#rhs1 = 40*nu*x**2*y**3*sin(t) - 60*nu*x**2*y**2*sin(t) + 24*nu*x**2*y*(x - 1)**2*sin(t) + 20*nu*x**2*y*sin(t) - 12*nu*x**2*(x - 1)**2*sin(t) - 32*nu*x*y**3*sin(t) + 48*nu*x*y**2*sin(t) - 16*nu*x*y*sin(t) + 8*nu*y**3*(x - 1)**2*sin(t) - 12*nu*y**2*(x - 1)**2*sin(t) + 4*nu*y*(x - 1)**2*sin(t) - 4*x**3*y**2*(x - 1)**3*(2*x - 1)*(y - 1)**2*(2*y*(y - 1) + y*(2*y - 1) + (y - 1)*(2*y - 1) - 2*(2*y - 1)**2)*sin(t)**2 - 4*x**2*y**3*(x - 1)**2*cos(t) + 6*x**2*y**2*(x - 1)**2*cos(t) - 2*x**2*y*(x - 1)**2*cos(t) + 2*x*y**2*sin(t) - 2*x*y*sin(t) - y**2*sin(t) + y*sin(t) 
#
#rhs2 = -40*nu*x**3*y**2*sin(t) + 32*nu*x**3*y*sin(t) - 8*nu*x**3*(y - 1)**2*sin(t) + 60*nu*x**2*y**2*sin(t) - 48*nu*x**2*y*sin(t) + 12*nu*x**2*(y - 1)**2*sin(t) - 24*nu*x*y**2*(y - 1)**2*sin(t) - 20*nu*x*y**2*sin(t) + 16*nu*x*y*sin(t) - 4*nu*x*(y - 1)**2*sin(t) + 12*nu*y**2*(y - 1)**2*sin(t) + 4*x**3*y**2*(y - 1)**2*cos(t) - 4*x**2*y**3*(x - 1)**2*(y - 1)**3*(2*y - 1)*(2*x*(x - 1) + x*(2*x - 1) + (x - 1)*(2*x - 1) - 2*(2*x - 1)**2)*sin(t)**2 - 6*x**2*y**2*(y - 1)**2*cos(t) + 2*x**2*y*sin(t) - x**2*sin(t) + 2*x*y**2*(y - 1)**2*cos(t) - 2*x*y*sin(t) + x*sin(t) 
#
#rhs3 = 0 
rhs1p = 40*nu*pow(x,2)*pow(y,3)*sin(om*t) - 60*nu*pow(x,2)*pow(y,2)*sin(om*t) + 24*nu*pow(x,2)*y*pow((x - 1),2)*sin(om*t) + 20*nu*pow(x,2)*y*sin(om*t) - 12*nu*pow(x,2)*pow((x - 1),2)*sin(om*t) - 32*nu*x*pow(y,3)*sin(om*t) + 48*nu*x*pow(y,2)*sin(om*t) - 16*nu*x*y*sin(om*t) + 8*nu*pow(y,3)*pow((x - 1),2)*sin(om*t) - 12*nu*pow(y,2)*pow((x - 1),2)*sin(om*t) + 4*nu*y*pow((x - 1),2)*sin(om*t) - 4*pow(x,3)*pow(y,2)*pow((x - 1),3)*(2*x - 1)*pow((y - 1),2)*(2*y*(y - 1) + y*(2*y - 1) + (y - 1)*(2*y - 1) - 2*pow((2*y - 1),2))*pow(sin(om*t),2) - 4*pow(x,2)*pow(y,3)*pow((x - 1),2)*om*cos(om*t) + 6*pow(x,2)*pow(y,2)*pow((x - 1),2)*om*cos(om*t) - 2*pow(x,2)*y*pow((x - 1),2)*om*cos(om*t) + 2*x*pow(y,2)*sin(om*t) - 2*x*y*sin(om*t) - pow(y,2)*sin(om*t) + y*sin(om*t) 
#
rhs2p = -40*nu*pow(x,3)*pow(y,2)*sin(om*t) + 32*nu*pow(x,3)*y*sin(om*t) - 8*nu*pow(x,3)*pow((y - 1),2)*sin(om*t) + 60*nu*pow(x,2)*pow(y,2)*sin(om*t) - 48*nu*pow(x,2)*y*sin(om*t) + 12*nu*pow(x,2)*pow((y - 1),2)*sin(om*t) - 24*nu*x*pow(y,2)*pow((y - 1),2)*sin(om*t) - 20*nu*x*pow(y,2)*sin(om*t) + 16*nu*x*y*sin(om*t) - 4*nu*x*pow((y - 1),2)*sin(om*t) + 12*nu*pow(y,2)*pow((y - 1),2)*sin(om*t) + 4*pow(x,3)*pow(y,2)*pow((y - 1),2)*om*cos(om*t) - 4*pow(x,2)*pow(y,3)*pow((x - 1),2)*pow((y - 1),3)*(2*y - 1)*(2*x*(x - 1) + x*(2*x - 1) + (x - 1)*(2*x - 1) - 2*pow((2*x - 1),2))*pow(sin(om*t),2) - 6*pow(x,2)*pow(y,2)*pow((y - 1),2)*om*cos(om*t) + 2*pow(x,2)*y*sin(om*t) - pow(x,2)*sin(om*t) + 2*x*pow(y,2)*pow((y - 1),2)*om*cos(om*t) - 2*x*y*sin(om*t) + x*sin(om*t) 


# check the replacement ** by pow
Subdict = {x:0.1 , y:0.51, nu:3, t:5, om:2 }
print (rhs1 + ad1).subs(Subdict)
print rhs1p.subs(Subdict)
print (rhs2 + ad2).subs(Subdict)
print rhs2p.subs(Subdict)


