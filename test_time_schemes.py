from dolfin import *

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import os, glob

import dolfin_to_nparrays as dtn
import time_int_schemes as tis
import smartminex_tayhoomesh 

class TimestepParams(object):
	def __init__(self, method, N):
		self.t0 = 0
		self.tE = 1.0
		self.Omega = 2
		self.Ntslist = [32]
		self.NOutPutPts = 32
		self.method = method
		self.SadPtPrec = True
		self.UpFiles = UpFiles(method)
		self.Residuals = NseResiduals()
		self.linatol = 1e-4 # 0 for direct sparse solver
		self.TolCor = []
		self.MaxIter = None
		self.Ml = None  #preconditioners
		self.Mr = None
		self.ParaviewOutput = False
		self.SaveIniVal = False
		self.SaveTStps = False
		self.UsePreTStps = False
		self.TolCorB = True

def solve_stokesTimeDep(method=None, Omega=None, tE=None, Prec=None, N=None, NtsList=None, LinaTol=None, MaxIter=None, UsePreTStps=None, SaveTStps=None, SaveIniVal=None ):
	"""system to solve
	
  	 	 du\dt - lap u + grad p = fv
		         div u          = fp
	
	"""

	if N is None:
		N = 20 

	if method is None:
		method = 2
	
	if Omega is None:
		Omega = 3

	methdict = {
			1:'HalfExpEulSmaMin',
			2:'HalfExpEulInd2',
			3:'Heei2Ra'}

	# instantiate object containing mesh, V, Q, rhs, velbcs, invinds
	PrP = ProbParams(N,Omega)
	# instantiate the Time Int Parameters
	TsP = TimestepParams(methdict[method], N)

	if NtsList is not None:
		TsP.Ntslist = NtsList
	if LinaTol is not None:
		TsP.linatol = LinaTol
	if MaxIter is not None:
		TsP.MaxIter = MaxIter
	if tE is not None: 
		TsP.tE = tE
	if Omega is not None:
		TsP.Omega = Omega
	if SaveTStps is not None:
		TsP.SaveTStps = SaveTStps
	if UsePreTStps is not None:
		TsP.UsePreTStps = UsePreTStps
	if SaveIniVal is not None:
		TsP.SaveIniVal = SaveIniVal


	print 'Mesh parameter N = %d' % N
	print 'Time interval [%d,%1.2f]' % (TsP.t0, TsP.tE)
	print 'Omega = %d' % TsP.Omega
	print 'You have chosen %s for time integration' % methdict[method]
	print 'The tolerance for the linear solver is %e' %TsP.linatol

	# get system matrices as np.arrays
	Ma, Aa, BTa, Ba, MPa = dtn.get_sysNSmats(PrP.V, PrP.Q)
	fv, fp = dtn.setget_rhs(PrP.V, PrP.Q, PrP.fv, PrP.fp)

	# condense the system by resolving the boundary values
	Mc, Ac, BTc, Bc, fvbc, fpbc, bcinds, bcvals, invinds = \
			dtn.condense_sysmatsbybcs(Ma,Aa,BTa,Ba,fv,fp,PrP.velbcs)
	
	if method != 2:
		# Rearrange the matrices and rhs
		from smamin_utils import col_columns_atend
		from scipy.io import loadmat

		MSmeCL, BSme, B2Inds, B2BoolInv, B2BI = smartminex_tayhoomesh.get_smamin_rearrangement(N,PrP,Mc,Bc)

		FvbcSme = np.vstack([fvbc[~B2BoolInv,],fvbc[B2BoolInv,]])
		FpbcSme = fpbc

		PrP.Pdof = 0 # Thats how the smamin is constructed

		# inivalue 
		dname = 'IniValSmaMinN%s' % N
		try:
			IniV = loadmat(dname)
			qqpq_init = IniV['qqpq_old']
			vp_init = None
		except IOError:
			qqpq_init = None
	
	### Output
	try:
		os.chdir('json')
	except OSError:
		raise Warning('need "json" subdirectory for storing the data')
	os.chdir('..')

	if TsP.ParaviewOutput:
		os.chdir('results/')
		for fname in glob.glob(TsP.method + '*'):
			os.remove( fname )
		os.chdir('..')

	###
	# Time stepping
	###
	# starting value
	dimredsys = len(fvbc)+len(fp)-1
	vp_init   = np.zeros((dimredsys,1))
	
	for i, CurNTs in enumerate(TsP.Ntslist):
		TsP.Nts = CurNTs

		if method == 2:
			tis.halfexp_euler_nseind2(Mc,MPa,Ac,BTc,Bc,fvbc,fpbc,
					vp_init,PrP,TsP)
		elif method == 1:
			tis.halfexp_euler_smarminex(MSmeCL,BSme,MPa,FvbcSme,FpbcSme,
					B2BoolInv,PrP,TsP,vp_init,qqpq_init=qqpq_init)
		elif method == 3:
			tis.halfexp_euler_ind2ra(MSmeCL,BSme,MPa,FvbcSme,FpbcSme,
					vp_init,B2BoolInv,PrP,TsP)

		# Output only in first iteration!
		TsP.ParaviewOutput = False
	
	JsD = save_simu(TsP, PrP)
		
	return 

def plot_errs_res(TsP):

	plt.close('all')
	for i in range(len(TsP.Ntslist)):
		fig1 = plt.figure(1)
		plt.plot(TsP.Residuals.ContiRes[i])
		plt.title('Lina residual in the continuity eqn')
		fig2 = plt.figure(2)
		plt.plot(TsP.Residuals.VelEr[i])
		plt.title('Error in the velocity')
		fig3 = plt.figure(3)
		plt.plot(TsP.Residuals.PEr[i])
		plt.title('Error in the pressure')

	plt.show(block=False)

	return


def plot_exactsolution(PrP,TsP):

	u_file = File("results/exa_velocity.pvd")
	p_file = File("results/exa_pressure.pvd")
	for tcur in np.linspace(TsP.t0,TsP.tE,11):
		PrP.v.t = tcur
		PrP.p.t = tcur
		vcur = project(PrP.v,PrP.V)
		pcur = project(PrP.p,PrP.Q)
		u_file << vcur, tcur
		p_file << pcur, tcur


def setget_velbcs_zerosq(mesh, V):
	# Boundaries
	def top(x, on_boundary): 
		return  np.fabs(x[1] - 1.0 ) < DOLFIN_EPS 
			  # and (np.fabs(x[0]) > DOLFIN_EPS))
			  # and np.fabs(x[0] - 1.0) > DOLFIN_EPS )
			  

	def leftbotright(x, on_boundary): 
		return ( np.fabs(x[0] - 1.0) < DOLFIN_EPS 
				or np.fabs(x[1]) < DOLFIN_EPS 
				or np.fabs(x[0]) < DOLFIN_EPS)

	# No-slip boundary condition for velocity
	noslip = Constant((0.0, 0.0))
	bc0 = DirichletBC(V, noslip, leftbotright)

	# Boundary condition for velocity at the lid
	lid = Constant((0.0, 0.0))
	bc1 = DirichletBC(V, lid, top)

	# Collect boundary conditions
	velbcs = [bc0, bc1]

	return velbcs

def save_simu(TsP, PrP):
	import json
	DictOfVals = {'SpaceDiscParam': PrP.N,
			'Omega': PrP.omega,
			'TimeInterval':[TsP.t0,TsP.tE],
			'TimeDiscs': TsP.Ntslist,
			'LinaTol': TsP.linatol,
			'TimeIntMeth': TsP.method,
			'ContiRes': TsP.Residuals.ContiRes,
			'VelEr': TsP.Residuals.VelEr,
			'PEr': TsP.Residuals.PEr,
			'TolCor': TsP.TolCor}

	JsFile = 'json/Omeg%dTol%0.2eNTs%dto%dMesh%d' % (DictOfVals['Omega'], TsP.linatol, TsP.Ntslist[0], TsP.Ntslist[-1], PrP.N) +TsP.method + '.json'

	f = open(JsFile, 'w')
	f.write(json.dumps(DictOfVals))

	print 'For the error plot, run\nimport plot_utils as plu\nplu.jsd_plot_errs("' + JsFile + '")'

	return 


class ProbParams(object):
	def __init__(self,N,Omega):

		self.mesh = smartminex_tayhoomesh.getmake_mesh(N)
		self.N = N
		self.V = VectorFunctionSpace(self.mesh, "CG", 2)
		self.Q = FunctionSpace(self.mesh, "CG", 1)
		self.velbcs = setget_velbcs_zerosq(self.mesh, self.V)
		self.Pdof = 0  #dof removed in the p approximation
		self.omega = Omega
		self.nu = 0
		self.fp = Constant((0))
		self.fv = Expression(("40*nu*pow(x[0],2)*pow(x[1],3)*sin(omega*t) - 60*nu*pow(x[0],2)*pow(x[1],2)*sin(omega*t) + 24*nu*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*sin(omega*t) + 20*nu*pow(x[0],2)*x[1]*sin(omega*t) - 12*nu*pow(x[0],2)*pow((x[0] - 1),2)*sin(omega*t) - 32*nu*x[0]*pow(x[1],3)*sin(omega*t) + 48*nu*x[0]*pow(x[1],2)*sin(omega*t) - 16*nu*x[0]*x[1]*sin(omega*t) + 8*nu*pow(x[1],3)*pow((x[0] - 1),2)*sin(omega*t) - 12*nu*pow(x[1],2)*pow((x[0] - 1),2)*sin(omega*t) + 4*nu*x[1]*pow((x[0] - 1),2)*sin(omega*t) - 4*pow(x[0],3)*pow(x[1],2)*pow((x[0] - 1),3)*(2*x[0] - 1)*pow((x[1] - 1),2)*(2*x[1]*(x[1] - 1) + x[1]*(2*x[1] - 1) + (x[1] - 1)*(2*x[1] - 1) - 2*pow((2*x[1] - 1),2))*pow(sin(omega*t),2) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*omega*cos(omega*t) + 6*pow(x[0],2)*pow(x[1],2)*pow((x[0] - 1),2)*omega*cos(omega*t) - 2*pow(x[0],2)*x[1]*pow((x[0] - 1),2)*omega*cos(omega*t) + 2*x[0]*pow(x[1],2)*sin(omega*t) - 2*x[0]*x[1]*sin(omega*t) - pow(x[1],2)*sin(omega*t) + x[1]*sin(omega*t)", "-40*nu*pow(x[0],3)*pow(x[1],2)*sin(omega*t) + 32*nu*pow(x[0],3)*x[1]*sin(omega*t) - 8*nu*pow(x[0],3)*pow((x[1] - 1),2)*sin(omega*t) + 60*nu*pow(x[0],2)*pow(x[1],2)*sin(omega*t) - 48*nu*pow(x[0],2)*x[1]*sin(omega*t) + 12*nu*pow(x[0],2)*pow((x[1] - 1),2)*sin(omega*t) - 24*nu*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*sin(omega*t) - 20*nu*x[0]*pow(x[1],2)*sin(omega*t) + 16*nu*x[0]*x[1]*sin(omega*t) - 4*nu*x[0]*pow((x[1] - 1),2)*sin(omega*t) + 12*nu*pow(x[1],2)*pow((x[1] - 1),2)*sin(omega*t) + 4*pow(x[0],3)*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) - 4*pow(x[0],2)*pow(x[1],3)*pow((x[0] - 1),2)*pow((x[1] - 1),3)*(2*x[1] - 1)*(2*x[0]*(x[0] - 1) + x[0]*(2*x[0] - 1) + (x[0] - 1)*(2*x[0] - 1) - 2*pow((2*x[0] - 1),2))*pow(sin(omega*t),2) - 6*pow(x[0],2)*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) + 2*pow(x[0],2)*x[1]*sin(omega*t) - pow(x[0],2)*sin(omega*t) + 2*x[0]*pow(x[1],2)*pow((x[1] - 1),2)*omega*cos(omega*t) - 2*x[0]*x[1]*sin(omega*t) + x[0]*sin(omega*t)"), t=0, nu=self.nu, omega = self.omega )

		self.v = Expression((
			"sin(omega*t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)", 
			"sin(omega*t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), omega = self.omega, t = 0)
		self.vdot = Expression((
			"omega*cos(omega*t)*x[0]*x[0]*(1 - x[0])*(1 - x[0])*2*x[1]*(1 - x[1])*(2*x[1] - 1)",
			"omega*cos(omega*t)*x[1]*x[1]*(1 - x[1])*(1 - x[1])*2*x[0]*(1 - x[0])*(1 - 2*x[0])"), omega = self.omega, t = 0)
		self.p =  Expression(("sin(omega*t)*x[0]*(1-x[0])*x[1]*(1-x[1])"), omega = self.omega, t = 0)

		bcinds = []
		for bc in self.velbcs:
			bcdict = bc.get_boundary_values()
			bcinds.extend(bcdict.keys())

		# indices of the inner velocity nodes
		self.invinds = np.setdiff1d(range(self.V.dim()),bcinds)

class NseResiduals(object):
	def __init__(self):
		self.ContiRes = []
		self.VelEr = []
		self.PEr = []

class UpFiles(object):
	def __init__(self, name=None):
		if name is not None:
			self.u_file = File("results/%s_velocity.pvd" % name)
			self.p_file = File("results/%s_pressure.pvd" % name)
		else:
			self.u_file = File("results/velocity.pvd")
			self.p_file = File("results/pressure.pvd")


if __name__ == '__main__':
	solve_stokesTimeDep()
