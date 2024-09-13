 	#Zach Johnson 2022
# Neutral Pseudo Atom code

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres, LinearOperator, eigs, spsolve
from scipy.linalg import solve_banded
from scipy.optimize import root, newton_krylov, fsolve
from scipy.special import gammaincc, gamma


import os
from .config import CORE_DIR, PACKAGE_DIR

from hnc.hnc.hnc import Integral_Equation_Solver as IET
from hnc.hnc.constants import *

from .grids import NonUniformGrid
from .solvers import jacobi_relaxation, sor, gmres_ilu, tridiagsolve, Ndiagsolve
from .physics import FermiDirac, ThomasFermi, χ_Lindhard, χ_TF, More_TF_Zbar, Fermi_Energy, n_from_rs, Debye_length

import matplotlib.pyplot as plt
from matplotlib import colors

import json

import pandas as pd


class Atom():
	"""
	Basics of an Atom 
	"""
	def __init__(self, Z, A, name=''):
		self.A = A
		self.Z = Z
		self.name = name	

class AverageAtom(Atom):
	"""
	A NeutralPseudoAtom class
	"""
	def __init__(self, Z, A, Ti, Te, rs, R, initialize=True, μ_init = None, Zstar_init = 'More', 
		rmin=2e-2, Npoints=500, iet_R_over_rs = None, iet_N_bins = 2000, name='', ignore_vxc=False, xc_type='KSDT', fixed_Zstar = False, use_full_ne_for_nf=False,
		χ_type = 'Lindhard', χ_LFC=True, χ_finite_T=True,  gii_init_type = 'step', grid_spacing='quadratic', N_stencil_oneside = 2, savefolder = os.path.join(PACKAGE_DIR,"data")):
		super().__init__(Z, A, name=name)
		"""
		Generates an average atom (rs=R) or neutral pseudo atom (R>>r).
		Args:
			str gradient_correction: None means TF, 'W' means Weizacker, 'K' means Kirznihtz
			float Weizsacker_λ: λ=1 is traditional Weizsacker, λ=1/9th is traditional TF gradient correction 
		"""
		print("________________________")
		print("Generating NPA")
		self.Te = Te
		self.Ti = Ti

		self.rs = rs
		self.R = R

		if iet_R_over_rs is None:
			self.iet_R_over_rs = R/rs
		else:
			self.iet_R_over_rs = iet_R_over_rs # How far to go for integral equation part # How far to go for integral equation part
		self.iet_N_bins = iet_N_bins

		self.WSvol = 4/3*π*self.rs**3
		self.Vol   = 4/3*π*self.R**3
		self.gii_init_type = gii_init_type 

		# Exchange-Correlation handling
		self.TF = ThomasFermi(self.Te, ignore_vxc = ignore_vxc, xc_type=xc_type )
		self.ignore_vxc = ignore_vxc
		self.xc_type   = xc_type 
		self.vxc_f = self.TF.vxc_func

		self.μ_init = μ_init
		
		if Zstar_init =='More':
			self.Zstar_init = More_TF_Zbar(self.Z, n_from_rs(rs), self.Te )
			print(f"Using More TF fit for initial Zstar = {self.Zstar_init:0.3f}")
		else:
			self.Zstar_init = Zstar_init

		self.fixed_Zstar = fixed_Zstar # If True, use Zstar_init always, don't update
		self.use_full_ne_for_nf = use_full_ne_for_nf
		self.χ_type = χ_type
		self.χ_LFC = χ_LFC
		self.χ_finite_T = χ_finite_T

		# Instantiate 1-D grid and ThomasFermi
		print("	Intializing grid")
		self.N_stencil_oneside = N_stencil_oneside 
		self.grid = NonUniformGrid(rmin, R, Npoints, self.rs, spacing=grid_spacing, N_stencil_oneside=N_stencil_oneside)
		self.rws_index = np.argmin(np.abs(self.grid.xs - self.rs))
		self.make_fast_n_TF()


		print(f"Initializing, {initialize}")
		if initialize:
			# Initializing densities and potentials
			print("Initializing")
			self.reinitialize()
			print("Initialized")
			self.gii_initial = self.gii.copy()

		self.savefolder = savefolder

	@property
	def savefolder(self):
		return self._savefolder
	
	@savefolder.setter
	def savefolder(self, folder):
		if not os.path.exists(folder):
			os.makedirs(folder)
		self._savefolder = folder


	def make_fast_n_TF(self):
		φion = self.Z/self.grid.xs - self.Z/self.grid.xmax # making sure handling this part. before initialization
		max_η = np.max(1.2*φion/self.Te)
		etas = np.sort(np.concatenate([np.geomspace(1e-8,max_η,num=10000),-np.geomspace(1e-8,10**(2.5),num=5000)]))
		I12_values = FermiDirac.Ionehalf(etas)
		Ionehalf = interp1d(etas, I12_values, kind='linear', bounds_error=False, fill_value=(0, None))
		self.fast_n_TF = lambda eta: (np.sqrt(2)/np.pi**2)*self.Te**(3/2)*Ionehalf(eta)
		# self.fast_n_TF = lambda eta: (np.sqrt(2)/np.pi**2)*self.Te**(3/2)*FermiDirac.Ionehalf(eta)


	def interp_to_grid(self, r_data, f_data):
		logr_data = np.log(r_data)
		logf_data = np.where(f_data==0, -1e1, np.log(f_data) )
		f = interp1d(logr_data, logf_data, bounds_error=False, fill_value = (logf_data[0], logf_data[-1]) )
		new_logf_data = f(np.log(self.grid.xs))
		return np.exp(new_logf_data) 

	def print_metric_units(self):
		aBohr = 5.29177210903e-9 # cm
		print("\n_____________________________________\nPlasma Description in A.U. and metric")
		print("n_ion: {0:10.3e} [A.U.], {1:10.3e} [1/cc]".format(self.ni_bar, self.ni_bar/aBohr**3))
		print("T:     {0:10.3e} [A.U.], {1:10.3e} [eV]".format(self.Te, self.Te*AU_to_eV ))
		print("\n")

	### Saving data
	def save_data(self):
		# Electron File
		err_info = f"# Convergence: Err(φ)={self.poisson_err:.3e}, Err(n_e)={self.rho_err:.3e}, Err(IET)={self.iet.final_Picard_err:.3e}, Q_net={self.Q:.3e}\n"
		aa_info = '# {{"name":"{0}", "Z":{1}, "Zstar":{2}, "A":{3},  "μ[AU]": {4:.10e}, "Te[AU]": {5:.3e}, "Ti[AU]": {6:.3e}, "rs[AU]": {7:.3e} }}\n'.format(self.name, self.Z, self.Zstar, self.A, self.μ, self.Te,self.Ti, self.rs)
		column_names = f"   {'r[AU]':15} {'n[AU]':15} {'nf[AU]':15} {'nb[AU]':15} {'n_ion[AU]':15} {'φtot[AU]':15} {'δVxc/δρ[Au]':15} {'U_ei[AU]':15} {'U_ii[AU]':15} {'g_ii':15} "
		header = ("# All units in Hartree [AU] if not specified\n"+
			    err_info + aa_info + column_names)   
		data = np.array([self.grid.xs, self.ne, self.nf, self.nb, self.ni, self.φe + self.φion, self.vxc_f(self.ne), self.Uei, self.uii_eff, self.gii_from_iet] ).T
		
		txt='{0}_{1}_R{2:.1e}_rs{3:.1e}_Te{4:.1e}eV_Ti{5:.1e}eV_electron_info.dat'.format(self.name, self.aa_type, self.R, self.rs, self.Te*AU_to_eV, self.Ti*AU_to_eV, self.Zstar)
		self.savefile = os.path.join(self.savefolder,txt)
		np.savetxt(self.savefile, data, delimiter = ' ', header=header, fmt='%15.6e', comments='')

		# Ion file
		err_info = f"# Convergence: Err(φ)={self.poisson_err:.3e}, Err(n_e)={self.rho_err:.3e}, Err(IET)={self.iet.final_Picard_err:.3e}, Q_net={self.Q:.3e}\n"
		aa_info = '# {{"name":"{0}", "Z":{1}, "Zstar":{2}, "A":{3},  "μ[AU]": {4:.10e}, "Te[AU]": {5:.3e}, "Ti[AU]": {6:.3e}, "rs[AU]": {7:.3e} }}\n'.format(self.name, self.Z, self.Zstar, self.A, self.μ, self.Te, self.Ti, self.rs)
		column_names = f"   {'r[AU]':15} {'U_ei[AU]':15} {'U_ii[AU]':15} {'g_ii':15} "
		header = ("# All units in Hartree [AU] if not specified\n"+
			    err_info + aa_info + column_names)   
		data = np.array([self.iet.r_array*self.rs, self.Uei_iet, self.iet.βu_r_matrix[0,0]*self.Ti , self.iet.h_r_matrix[0,0]+1 ] ).T
		
		txt='{0}_{1}_R{2:.1e}_rs{3:.1e}_Te{4:.1e}eV_Ti{5:.1e}eV_IET_info.dat'.format(self.name, self.aa_type, self.R, self.rs, self.Te*AU_to_eV, self.Ti*AU_to_eV, self.Zstar)
		self.savefile = os.path.join(self.savefolder,txt)
		np.savetxt(self.savefile, data, delimiter = ' ', header=header, fmt='%15.6e', comments='')

	def set_physical_params(self):
		self.ni_bar = 1/self.WSvol # Average Ion Density of plasma
		self.ne_bar = self.Zstar * self.ni_bar # Average Electron Density of plasma
	
		self.EF  = Fermi_Energy(self.ne_bar)
		self.kF = (2*self.EF)**(1/2)

		self.λTF = Debye_length(self.Te, self.ni_bar, self.Zstar)
		self.kTF = 1/self.λTF

		self.κ   = self.λTF*self.rs
		self.φ_κ_screen = 3/self.rs#self.κ
		self.Γ   = self.Zstar**2/(self.rs*self.Ti)
		self.make_χee()

	def reinitialize(self):
		self.Zstar  = self.Zstar_init

		# Initialize Physical Parameters		
		self.set_physical_params()

		# Initialize Ion density
		self.make_iet()
		self.make_gii()
		self.make_ρi()

		# Initialize chemical potential
		if self.μ_init==None:
			self.set_μ_infinite()
		else: 
			self.μ = self.μ_init

		# Initializing potentials
		self.φion = self.Z/self.grid.xs - self.Z/self.grid.xmax #1/r with zero at boundary 
		# self.φion = self.Z/self.grid.xs - self.Z/self.grid.xs[self.rws_index] #1/r with zero at boundary 
		self.φe_init = self.Z/self.grid.xs*np.exp(-self.kTF*self.grid.xs) - self.φion
		# self.φe_init = self.grid.ones.copy() #-self.φion #self.grid.zeros.copy()
		self.φe = self.φe_init.copy()
		

		# Initializing densities 
		self.initialize_ne()
		# self.set_μ_infinite()

		print("Intialized Potentials and Densities")

	### Initializing
	def initialize_ne(self):
		"""
		Initial Guess for electron charge densit using 
		"Approximate Solution of the Thomas–Fermi Equation for Free Positive Ions" 
			by Aleksey A. Mavrin and Alexander V. Demura 
		"""
		
		# Use approximate density based on plasma electron density, ignoring unknown W correction
		r_TF = 1/4 * ((9*π**2)/(2*self.Z))**(1/3)
		x = lambda r: r/r_TF
		Φ0_Mavrin_Demura = lambda x: ((1 + 1.81061 * x**(1/2) + 0.60112 * x) / (1 + 1.81061 * x**(1/2) + 1.39515 * x + 0.77112 * x**(3/2) + 0.21465 * x**2 + 0.04793 * x**(5/2)))**2
		z = lambda x: np.log(1+x) 
		η0_Mavrin_Demura = lambda x: np.exp(z(x) + 0.3837 * z(x)**2 + 0.0892 * z(x)**3 - 0.0170 * z(x)**4) - 1
		# q = More_TF_Zbar(self.Z, n_from_rs(self.rs), self.Te)/self.Z
		q = self.Zstar_init/self.Z
		x0_func = lambda q: 10.232/q**(1/3) * (1 - 0.917 * q**0.257) if q<=0.45 else 2.960* ((1-q)/q)**(2/3)
		x0 = x0_func(q)
		k = -Φ0_Mavrin_Demura(x0)/η0_Mavrin_Demura(x0)
		Φ_Mavrin_Demura = lambda x: Φ0_Mavrin_Demura(x) + k * η0_Mavrin_Demura(x)  
		nb_Mavrin_Demura_func = lambda r: np.nan_to_num(self.Z/(4*π*r_TF**3) * (Φ_Mavrin_Demura(x(r))/x(r) )**1.5)

		nb_Mavrin_Demura = nb_Mavrin_Demura_func(self.grid.xs)
		Nb_Mavrin_Demura = self.grid.integrate_f(nb_Mavrin_Demura)
		# nf_Mavrin_Demura = (self.Z - Nb_Mavrin_Demura)/self.Vol * np.ones_like(self.grid.xs)
		nf_Mavrin_Demura = self.ne_bar * np.ones_like(self.grid.xs)
		ne_Mavrin_Demura  = nb_Mavrin_Demura + nf_Mavrin_Demura 

		self.nb_init = nb_Mavrin_Demura
		self.nf_init = nf_Mavrin_Demura
		self.ne_init = ne_Mavrin_Demura

		self.nb = self.nb_init 
		self.nf = self.nf_init 
		self.ne = self.ne_init 

		self.set_μ_infinite()
	def get_βVeff(self, φe, ne, ne_bar):
		βVeff = ( -φe - self.φion + self.vxc_f(ne) - self.vxc_f(ne_bar) )/self.Te
		return βVeff
        
	# def get_βVeff(self, φe, ne, ne_bar):
	# 	if self.ignore_vxc and self.gradient_correction is None:
	# 		βVeff = ( -φe - self.φion )/self.Te
	# 	elif self.ignore_vxc==False and self.gradient_correction is None:
	# 		βVeff = ( -φe - self.φion + self.vxc_f(ne) - self.vxc_f(ne_bar) )/self.Te
	# 	elif self.ignore_vxc and self.gradient_correction is not None:
	# 		βVeff = ( -φe - self.φion + self.get_gradient_energy(ne))/self.Te
	# 	elif self.ignore_vxc==False and self.gradient_correction is not None:
	# 		βVeff = ( -φe - self.φion + self.get_gradient_energy(ne) + self.vxc_f(ne) - self.vxc_f(ne_bar))/self.Te

	# 	return βVeff



	def get_eta_from_sum(self, φe, ne, μ, ne_bar):
		βVeff = self.get_βVeff(φe, ne, ne_bar)
		eta = μ/self.Te - βVeff
		return eta

	def make_ne_TF(self):
		"""
		Sets e-density using self μ
		"""
		self.ne = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)

	def get_ne_TF(self, φe, ne, μ, ne_bar):
		"""
		Generates electron density self.ne_grid from fermi integral
		Args: 
			float μ: chemical potential
			float ne: 
		Returns:
			None
		"""		   
		eta = self.get_eta_from_sum(φe, ne, μ, ne_bar)
		ne = self.fast_n_TF( eta)
		return ne
	
	def get_Q(self):
		self.Qion = self.Z  + self.grid.integrate_f( self.ρi )
		return self.Qion - self.grid.integrate_f(self.ne)

	
	## Chemical Potential Methods 
	def get_μ_neutral(self, μ_guess = None):
		"""
		Finds μ through enforcing charge neutrality
		"""
		end_index = None#self.rws_index
		min_μ = lambda μ: abs(  self.Z + self.grid.integrate_f( self.ρi - self.get_ne_TF(self.φe, self.ne, μ, self.ne_bar) , end_index=end_index) )**2

		if μ_guess is None:
			μ_guess = self.μ
		root_and_info = root(min_μ, μ_guess, tol=1e-12)
		μ = root_and_info['x'][0]
		return μ

	def set_μ_neutral(self):
		self.μ = self.get_μ_neutral()

	def get_μ_infinite(self):
		# Set μ such that at R, get asymptotic ne_bar right
		eta_bar = self.TF.η_interp(self.ne_bar)
		μ_init  = eta_bar*self.Te #+ self.vxc_f(self.ne_bar)  
		μ  = μ_init  
		return μ

	def set_μ_infinite(self):
		self.μ = self.get_μ_infinite()

	def update_μ_newton(self, alpha1=1e-2, alpha2= 1e-2):
		"""
		Finds μ through enforcing charge neutrality,using I_1/2 n, NOT guaranteed neutral self.ne 
		"""
		ne = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)
		Qnet = self.Qion - self.grid.integrate_f(ne)#self.get_Q()
		eta = self.get_eta_from_sum(self.φe, ne, self.μ, self.ne_bar)
		nprime = self.TF.n_TF_prime(self.Te,eta)
		
		self.μ +=  alpha1 * self.Te*Qnet/np.sqrt(self.grid.integrate_f(nprime)**2 + 0.1)
			
	## IONS
	def make_iet(self):
		print("	Creating Integral Equation Solver")
		self.set_physical_params()

		self.iet = IET(1, self.Γ, 3/(4*π), self.Ti, 1, kappa=self.κ,
		  R_max = self.iet_R_over_rs, N_bins=self.iet_N_bins, dst_type=3)

	def make_χee(self):

		if self.χ_type == 'Lindhard':
			self.χee_func = lambda k: χ_Lindhard(self.μ, self.Te, k, self.kF, LFC=self.χ_LFC, finite_T=self.χ_finite_T)
			self.χee_func = np.vectorize(self.χee_func)
		if self.χ_type == 'TF':
			self.χee_func = lambda k: χ_TF(k, self.kF)
			self.χee_func = np.vectorize(self.χee_func)

	def make_Uei(self):
		if self.use_full_ne_for_nf == True:
			nf = self.ne.copy()
			nf = np.where(nf<=1e-30, 1e-30, nf)
		else:
			nf = self.nf.copy() #+ self.ni_bar
			nf = np.where(nf<=1e-30, 1e-30, nf)

		etas = self.TF.η_interp(nf) # η = β( μ + self.φe + self.φion - self.vxc_f(ne)  )         
		if self.ignore_vxc:
			totφ_pseudo = etas*self.Te - self.μ # total potential that must be acting on nf
		else:
			totφ_pseudo = etas*self.Te + self.vxc_f(nf) - self.vxc_f(self.ne_bar)  - self.μ # total potential that must be acting on nf
		φe_from_nf, _ = self.get_φe( (-nf + self.ρi)  ) # potential from nf itself

		φ_pseudo = (totφ_pseudo - φe_from_nf)
		self.Uei = -(φ_pseudo + self.Zstar/self.R)

	def make_Uei_iet(self):
		self.make_Uei()
		Uei_iet_interp = interp1d(self.grid.xs, self.Uei , bounds_error=False, fill_value='extrapolate')
		
		@np.vectorize
		def get_fixed_Uei(r):
		    if r<self.grid.xs[0]:
		        return self.Uei[0]
		    elif r>self.grid.xs[-1]:
		        return -self.Zstar/r
		    else:
		        return Uei_iet_interp(r)

		self.Uei_iet = get_fixed_Uei(self.iet.r_array*self.rs)
		self.Uei_iet_k = self.rs**3 * self.iet.FT_r_2_k( self.Uei_iet)


	def set_uii_eff(self):
		self.χee_iet = self.χee_func(self.iet.k_array/self.rs)
		self.set_physical_params()
		self.make_Uei_iet()
		
		u_k_Y_approx = 4*π*self.Zstar**2/(  (self.iet.k_array/self.rs)**2 + (self.κ/self.rs)**2)
		u_r_Y_approx = self.Zstar**2/(self.iet.r_array*self.rs)*np.exp(-self.κ*self.iet.r_array)

		self.uii_k_eff_iet = 4*π*self.Zstar**2/(self.iet.k_array/self.rs)**2 + self.χee_iet*self.Uei_iet_k**2 - u_k_Y_approx
		self.uii_r_eff_iet = self.iet.FT_k_2_r(self.uii_k_eff_iet*self.rs**-3) + u_r_Y_approx
		self.uii_eff = interp1d(self.iet.r_array*self.rs, self.uii_r_eff_iet, bounds_error=False, fill_value='extrapolate')(self.grid.xs)

		self.iet.set_βu_matrix( np.array([[self.uii_r_eff_iet]])/self.Ti )

		self.solve_iet()

	def solve_iet(self, **kwargs):
		print("	-------------------")
		print("	Solving IET.")
		my_kwargs = {'iters_to_wait':1e5, 'tol':1e-12, 'verbose':False,'num_iterations':1e4}
		my_kwargs.update(kwargs)
		self.iet.HNC_solve(**my_kwargs)
		self.gii_from_iet = interp1d(self.iet.r_array*self.rs, self.iet.h_r_matrix[0,0] + 1, bounds_error=False, fill_value='extrapolate', kind='cubic')(self.grid.xs)

	def make_gii(self, **kwargs):
		"""
		Initially just simple step function. Later HNC will be used.
		"""
		if self.rs == self.R:
			self.gii = np.zeros_like(self.grid.xs)
		elif self.gii_init_type == 'iet':
			self.solve_iet(**kwargs)	
			gii_iet = self.iet.h_r_matrix[0,0] + 1
			self.gii = interp1d(self.iet.r_array*self.rs, gii_iet, bounds_error=False, fill_value='extrapolate')(self.grid.xs)
		else: #self.gii_init_type == 'step'
			self.gii = np.ones_like(self.grid.xs) * np.heaviside(self.grid.xs - self.rs, 1)

	def make_ρi(self):
		"""
		Ions other than the central ion. The plasma ions based on gii(r).
		"""
		self.ni = self.ni_bar * self.gii # Number density
		self.ρi = self.ni * self.Zstar   # Charge density
		self.Qion = self.Z  + self.grid.integrate_f( self.ρi )
		# print("Qion = {}".format(self.Qion))

	def get_φe(self, ρ):
		"""
		5 diagonal
		Use solve_banded to solve Poisson Equation for φe using charge density ρ, which might be ρ = self.ρi - self.ne for example 
		Returns:
			float err: residual Ax-b 
		"""
		
		def explicit_Ab(): #With BC at core and edge
			#### A ####
			### FIRST BULK VALUES ###
			A = -self.grid.A_laplace
			
			dx= self.grid.dx
			x = self.grid.xs
			### BOUNDARIES ###
			A[0,:] =  -self.grid.A_dfdx[0,:] # get electric field at origin 
			A[-1,:] *= 0 #Ensure boundary condition only at edge 
			A[-1,-1] = 1 # Sets φe[-1]=0
			# A[-1,-1] = 1#/dx[-1] # Sets grad φe[-1]=0
			# A[-1,-2] = -1/dx[-1] # Sets grad φe[-1]=0
			
			
			b = np.zeros(self.grid.Nx)
			b[0]    = 0 # 8*np.pi/9*ρ[0]*x[0] # 8*np.pi/9*ρ[0]*x[0]
			# b[0]    = -20*self.grid.vols[0]*ρ[0]/x[1]**2
			b[-1]  =  0#(+self.get_Q() + self.Z )/self.R**2 #sets φe[-1]=0
			b[1:-1]= 4*π*ρ[1:-1]

			return A, b

		# Use function to create A, b
		A, b = explicit_Ab()
		self.Ab = A, b

		φe = Ndiagsolve(A, b, self.N_stencil_oneside)
		# φe_3 = tridiagsolve(A, b) # only works as a check for N_stencil_oneside=1
		# self.φe = jacobi_relaxation(A, b, self.φe, nmax=200) # quick smoothing, not to convergence

		φe = φe - φe[-1]
		rel_errs = (np.abs(A @ φe - b)[:-1]/b[:-1])
		
		return φe, rel_errs

	def get_φe_screened(self, ρ):
		"""
		Use solve_banded to solve Poisson Equation for φe using charge density ρ, which might be ρ = self.ρi - self.ne for example 
		Screened version based on G. P. Kerker "Efficient iteration scheme for self-consistent pseudopotential calculations"	
		https://doi.org/10.1103/PhysRevB.23.3082
		
		(-Δ + κ^2) φ = + 4πρ  + κ^2 φ
		Returns:
			float err: residual Ax-b 
		"""
		
		def explicit_Ab(): #With BC at core and edge
			#### A ####
			### FIRST BULK VALUES ###
			A = -self.grid.A_laplace
			inner_diag_indices = ( np.arange(1, self.grid.Nx-1 ), np.arange(1, self.grid.Nx-1 ) )
			A[inner_diag_indices] += self.φ_κ_screen**2 * np.ones( self.grid.Nx-2 )

			dx= self.grid.dx
			x = self.grid.xs
			### BOUNDARIES ###
			A[0,:] =  -self.grid.A_dfdx[0,:] # get electric field at origin 
			A[-1,:] *= 0 #Ensure boundary condition only at edge 
			A[-1,-1] = 1 # Sets φe[-1]=0
			# A[-1,:] = self.grid.A_dfdx[-1,:]#1 # Sets φe[-1]=0

			
			b = np.zeros(self.grid.Nx)
			b[0]    = 0#*8*np.pi/9*ρ[0]*x[0]
			b[-1]  =  0#
			# b[-1]  =  -self.grid.A_dfdx[-1,:].dot(self.φion)
			b[1:-1]= 4*π*ρ[1:-1] + self.φe[1:-1] * self.φ_κ_screen**2 * np.ones( self.grid.Nx-2 )

			return A, b

		# Use function to create A, b
		A, b = explicit_Ab()
		self.Ab = A, b

		φe = Ndiagsolve(A, b, self.N_stencil_oneside)
		# self.φe = jacobi_relaxation(A, b, self.φe, nmax=200) # quick smoothing, not to convergence

		φe = φe - φe[-1]
		LHS = A @ φe
		RHS = b
		rel_errs = 0.5*(np.abs(LHS - RHS)[1:-1]/np.sqrt(LHS**2 + RHS**2)[1:-1])
		
		return φe, rel_errs
	

	# def get_ne_guess(self, **kwargs):
	# 	if self.gradient_correction is not None:
	# 		new_ne = self.get_ne_guess_W(**kwargs)
	# 	else:
	# 		new_ne = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)
	# 	return new_ne	

	def get_ne_picard(self, alpha, **kwargs):
		# if self.gradient_correction is not None:
		# 	# Above updates very slowly when gradients are very small, so we add in the update based on TF integral  	
		# 	ne_guess = self.get_ne_W(alpha )
		# else:
		ne_guess = self.get_ne_guess()#self.φe, self.ne, self.μ, self.ne_bar)
		ne_picard = (1-alpha)*self.ne + alpha*ne_guess
		return ne_picard 

	def update_ne_picard(self, alpha, **kwargs):
		self.ne = self.get_ne_picard(alpha)

	def make_bound_free(self):
		etas = self.TF.η_interp(self.ne)
		xmid = -self.get_βVeff(self.φe, self.ne, self.ne_bar)
		c = 0.05
		fcut = (1 + np.exp(-1/c))/( 1 + np.exp( ( self.grid.xs - self.rs)/(c*self.rs)) )

		self.nb = ThomasFermi.n_bound_TF(self.Te, etas, xmid )*fcut
		self.nf = self.ne - self.nb  
	
	def get_newton_change(self, x0, x1, y0, y1):
		dydx = (y1- y0)/(x1 - x0)
		Δx_21_guess = -y1/dydx # meaning, x2 = x1 + Δx_21_guess
		return Δx_21_guess

	def newton_update_from_guess(self, x0, x1, x1_guess_from_x0, x2_guess_from_x1, constrained=True, α=0.25):
		Δx0 = x1_guess_from_x0 - x0 # Idea is we want 
		Δx1 = x2_guess_from_x1 - x1
		newton_change = self.get_newton_change(x0, x1, Δx0, Δx1)
		if constrained == True:
			x2 = x1 + np.sign(newton_change) * np.min([np.abs(x1)*0.25, np.abs(newton_change)])
		else:
			x2 = x1 + newton_change
		# print(f"Old: ΔZ_new {ΔZ_new}, ΔZ_old {ΔZ_old}, dZguess_dZ {dZguess_dZ}, newton_change {newton_change}")
		return x2

	def update_ne_newton(self, alpha = 2e-1):
		"""
		Gets bound free separation using approximation  in ThomasFermi. 
		Only iteratively updates Zstar
		Returns: 
			Exact Zstar using bound, free.

		"""
		self.new_ne_guess = self.get_ne_guess()

		try: # Do newton
			new_ne = self.newton_update_from_guess(self.old_ne, self.ne, self.old_ne_guess, self.new_ne_guess, constrained=True)
			self.old_ne_guess = self.new_ne_guess.copy()
			self.old_ne = self.ne.copy()
			self.ne = new_ne

		except AttributeError:	 # Do Picard
			self.old_ne = self.ne.copy()
			self.ne = (1-alpha)*self.ne + alpha*self.new_ne_guess
		
		self.old_ne_guess = self.new_ne_guess.copy()

	def get_Zstar_from_selfnb(self):
		return self.Z - self.grid.integrate_f(self.nb)

	def set_Zstar(self):
		self.Zstar = self.get_Zstar_from_selfnb()

	def update_newton_Zstar(self, alpha = 2e-1):
		"""
		Gets bound free separation using approximation  in ThomasFermi. 
		Only iteratively updates Zstar
		Returns: 
			Exact Zstar using bound, free.

		"""
		self.make_bound_free()

		self.new_Zstar_guess = self.get_Zstar_from_selfnb()

		try:
			new_Zstar = self.newton_update_from_guess(self.old_Zstar, self.Zstar, self.old_Zstar_guess, self.new_Zstar_guess)
			self.old_Zstar = self.Zstar
			self.old_Zstar_guess = self.new_Zstar_guess
			self.Zstar = new_Zstar
			
		except AttributeError:	
			self.old_Zstar = self.Zstar
			self.Zstar = (1-alpha)*self.Zstar + alpha*(self.new_Zstar_guess-self.Zstar) #smaller nonzero update
		
		self.old_Zstar_guess = self.new_Zstar_guess

		if self.Zstar<=1e-6:
			self.Zstar = 1e-6

		#Update ne_bar, free density etc.
		self.set_physical_params() 
		self.make_ρi()

	def update_ρi_and_Zstar_to_make_neutral(self, alpha = 1):
		Zstar_needed_for_neutral = (self.grid.integrate_f(self.ne)-self.Z)/self.grid.integrate_f(self.ni)
		self.Zstar = self.Zstar + alpha*(Zstar_needed_for_neutral - self.Zstar) #smaller nonzero update
		if self.Zstar<=1e-6:
			self.Zstar = 1e-6

		self.set_physical_params()
		self.make_ρi()
		return Zstar_needed_for_neutral

	def rel_error(self, vec_1, vec_2, weight=1, abs=False):
		if abs==False:
			return np.linalg.norm( weight*(vec_1 - vec_2)  )/np.linalg.norm( weight*np.sqrt(vec_1**2 + vec_2**2) )
		else:
			return np.linalg.norm( weight*np.abs(vec_1 - vec_2)  )/np.linalg.norm( weight*np.sqrt(vec_1**2 + vec_2**2) )

	def L2_change(self,new,old):	
		"""
		Weighted error metric for difference between new, old of multiple parameters.
		"""
		N_params = 3
		coeffs = np.array([1,1,1])
		rel_err  = lambda a, b: abs(  (a-b)/np.sqrt(a**2+b**2) )
		rel_err = np.vectorize(rel_err)
		# print("		Errs: ", rel_err(new,old))
		return coeffs @ rel_err(new,old)
	
	def solve_TF(self, verbose=False, picard_alpha = 1e-2, nmax = 1e4, tol=1e-4, save_steps=False):
		"""
		Solve TF OFDFT equation, assuming a given Zbar for the plasma ions 
		"""
		if verbose:
			print("Beginning self-consistent electron solver.")
			print("_________________________________")
		
		self.ne_list = [self.ne.copy()]
		self.ρi_list = [self.ρi.copy()]
		self.ne_bar_list = [self.ne_bar]
		self.μ_list, self.rho_err_list, self.change_list = [self.μ], [0], [0]
		self.φe_list = [self.φe]
		self.new_Zstar_guess = self.Zstar
		Q = 0
		n = 0
		converged, μ_converged, Zbar_converged = False, False, False
		while not converged and n < int(nmax):
			Q_old = Q
			old = self.μ, np.mean(self.ne), np.mean(self.φe) 
			
			# Update physics in this order
			self.φe, poisson_err = self.get_φe_screened(self.ρi - self.ne)
			poisson_err = np.mean(poisson_err)
			self.update_ne_picard(alpha=picard_alpha)
			self.update_bulk_params(n)
			"""
			if not remove_ion: # Normal Route
				if self.rs==self.R: # IS model
					if self.fixed_Zstar == False and n>n_wait_update_Zstar:
						self.update_newton_Zstar() 			# get Zstar from bound/free
					if n%10==0 or n<5 and not μ_converged:
						self.set_μ_neutral()
					elif not μ_converged: 
						self.update_μ_newton(alpha1=1e-3)
				else: # CS model
					if self.fixed_Zstar == False and n>n_wait_update_Zstar:
						if  n%1==0 and not Zbar_converged:
							old_ne_bar = self.ne_bar
							self.update_newton_Zstar()
							self.ne   += self.ne_bar - old_ne_bar
					
					self.update_ρi_and_Zstar_to_make_neutral()
					self.set_μ_infinite()
					self.set_μ_neutral()	


			else: #Simulating removal of ion, keep μ the same.
				if n%10==0 or n<5 and not μ_converged:
					self.set_μ_neutral()
				elif not μ_converged: 
					self.update_μ_newton(alpha1=1e-3)
			"""
			# Convergence testing
			TF_ne = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)
			rho_err = self.rel_error(TF_ne, self.ne, weight=4*π*self.grid.xs**2, abs=True)

			# Save convergence variables
			Q = self.get_Q()
			delta_Q = self.get_Q() - Q_old

			new = self.μ, np.mean(self.ne), np.mean(self.φe) 
			change  = self.L2_change(new,old)
			self.μ_list.append(self.μ); self.rho_err_list.append(rho_err); self.change_list.append(change)
			self.ne_list.append(self.ne.copy())
			self.φe_list.append(self.φe.copy())
			self.ρi_list.append(self.ρi.copy())
			self.ne_bar_list.append(self.ne_bar)

			# μ and Zbar convergence tests
			if np.abs(1-self.μ/old[0])<1e-5:
				μ_converged = True
			else:
				μ_converged = False
			
			if np.abs(1-self.Zstar/self.new_Zstar_guess)<1e-5:
				Zbar_converged = True
			else:
				Zbar_converged = False

			if verbose and (n%25==0 or n<10):
				print("__________________________________________")
				print("TF Iteration {0}".format(n))
				print("	μ = {0:10.9e}, change: {1:10.9e} (converged={2})".format(self.μ, np.abs(1-self.μ/old[0]),μ_converged))	
				print("	φe Err = {0:10.3e}, φe change = {1:10.3e}".format(poisson_err, self.rel_error(self.φe_list[-1], self.φe_list[-2])  ))
				print("	ne Err = {0:10.3e}, ne change = {1:10.3e}".format(rho_err, self.rel_error(self.ne_list[-1], self.ne_list[-2])  ))
				print("	Q = {0:10.3e} -> {1:10.3e}, ".format(Q_old, Q))
				print("	Zstar guess = {0:10.3e}. Current Zstar: {1:10.3e} (converged={2})".format(self.new_Zstar_guess, self.Zstar, Zbar_converged))
				print("	Change = {0:10.3e}".format(change))

			# Converged ?
			# if remove_ion:
			# 	if  change<tol and abs(rho_err)<tol:
			# 		converged=True
			# else:
				# if abs(Q)<1e-3 and change<tol and abs(rho_err)<tol and μ_converged and Zbar_converged:
			if change<tol and abs(rho_err)<tol:# and μ_converged and Zbar_converged:
				converged=True
			n+=1

		print("__________________________________________")
		print("TF Iteration {0}".format(n))
		print("	μ = {0:10.9e}, change: {1:10.9e} (converged={2})".format(self.μ, np.abs(1-self.μ/old[0]),μ_converged))	
		print("	φe Err = {0:10.3e}, φe change = {1:10.3e}".format(poisson_err, self.rel_error(self.φe_list[-1], self.φe_list[-2])  ))
		print("	ne Err = {0:10.3e}, ne change = {1:10.3e}".format(rho_err, self.rel_error(self.ne_list[-1], self.ne_list[-2])  ))
		print("	Q = {0:10.3e} -> {1:10.3e}, ".format(Q_old, Q))
		print("	Zstar guess = {0:10.3e}. Current Zstar: {1:10.3e} (converged={2})".format(self.new_Zstar_guess, self.Zstar, Zbar_converged))
		print("	Change = {0:10.3e}".format(change))
		self.poisson_err = poisson_err
		self.rho_err = rho_err
		self.Q = Q
		return converged


	##############################	
	########## PLOTTING ##########
	##############################
	def get_Q_profile(self, ρ):
		return np.array([self.grid.integrate_f(ρ, end_index = index) for index in range(len(self.grid.xs))])


	def make_nice_text(self):
		text = ("{0}, {1}\n".format(self.name, self.aa_type.replace("_", " "))+ 
			r"$r_s$ = " + "{0:.2f},    ".format(self.rs,2) +
			r"$R_{NPA}$ = " + "{0:.2f}\n".format(self.R)  +
    			r"$T_e$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.Te, self.Te*AU_to_eV) +
    			r"$\mu$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.μ, self.μ*AU_to_eV) +
    			r"$\mu-V(R)$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.μ-self.vxc_f(self.ne)[-1],(self.μ-self.vxc_f(self.ne)[-1])*AU_to_eV) +
    			r"$Z^\ast =$ " + "{0:.2f}".format(self.Zstar)  )
		self.nice_text = text
		
	def convergence_plot(self, vars, var_names):
		fig, ax = plt.subplots(ncols=1, figsize=(7,5),facecolor='w')
		for var, name in zip(vars,var_names):
			ax.plot(var, '--.', label=name)

		ax.set_xlabel("Iteration",fontsize=20)
		# ax.set_ylabel("Error",fontsize=20)
		ax.tick_params(labelsize=20,which='both')
		#ax.set_yscale('symlog',linthresh=1e-3)
		ax.set_yscale('log')
		ax.set_ylim(np.max([1e-10, 0.1*np.min(vars)]),2*np.max(vars))
		ax.grid(which='both')
		ax.legend(fontsize=15)

		plt.tight_layout()

		#Save
		name = "NPA_convergence.png"
		plt.savefig(os.path.join(PACKAGE_DIR,"media",name), dpi=300, bbox_inches='tight',facecolor="w")
		#plt.show()
		return fig, ax


	def make_plots(self, show=False):

		fig, axs = plt.subplots(ncols=2,figsize=(20,8),facecolor='w')
  	      
  	      # Potential φ plot
		axs[0].plot(self.grid.xs , self.φion, label=r"$\phi_{ion}$")
		axs[0].plot(self.grid.xs , -self.φe, label=r"$-\phi_{e}$")
		axs[0].plot(self.grid.xs , self.φe + self.φion, label=r"$\phi$")
		# if not self.ignore_vxc:
		axs[0].plot(self.grid.xs , -self.vxc_f(self.ne) , label=r"$-v_{xc}[n_e]$")
		# if self.gradient_correction is not None:
		# 	axs[0].plot(self.grid.xs , -self.get_gradient_energy(self.ne) , label=r"$-v_{W}[n_e]$")
		axs[0].plot(self.grid.xs , -self.get_βVeff(self.φe, self.ne, self.ne_bar)*self.Te , label=r"$-V_{\rm eff}$")


		axs[0].set_ylabel(r'$\phi$ [A.U.]',fontsize=20)
		axs[0].set_ylim(-1e3,1e6)
		axs[0].set_yscale('symlog',linthresh=1e-8)

		# Density ne plot
		# axs[1].plot(self.petrov.r_data, self.petrov.rho_data + self.petrov.rho_0, 'k--', label="Petrov AA")
		axs[1].plot(self.grid.xs, self.ne , label=r'$n_e$')
		axs[1].plot(self.grid.xs, self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar) , label=r'$\frac{\sqrt{2}}{\pi^2}T^{3/2}\mathcal{I}_{1/2}(\eta)$')
		axs[1].plot(self.grid.xs, self.nb, label=r'$n_b$')
		axs[1].plot(self.grid.xs, self.nf, label=r'$n_f$')
		axs[1].plot(self.grid.xs, self.ρi, label=r'$ Z^\ast n^0_i g_{ii}(r) $ ')
		axs[1].plot(self.grid.xs, self.ρi - self.ne, label=r'$\Sigma_j \rho_j$ ')
		
		axs[1].set_ylabel(r'$n$ [A.U.]',fontsize=20)
		
		
		axs[1].set_ylim(-np.max(self.ne)/100,10*np.max(self.ne))
		#axs[1].set_yscale('symlog',linthresh=1e-3)
		axs[1].set_yscale('symlog',linthresh=self.ne_bar/1e8)
  
		for ax in axs:
			ax.set_xlim(self.grid.xs[0],self.grid.xs[-1])
			ax.set_xscale('log')
			ax.set_xlabel(r'$|r-R_1|$ [A.U.]',fontsize=20)
			ax.legend(loc="center left",fontsize=20,labelspacing = 0.1)
			ax.tick_params(labelsize=20)
			ax.grid(which='both',alpha=0.4)

			# make textbox
			text = ("{0}, {1}\n".format(self.name, self.aa_type.replace("_", " "))+ 
				r"$r_s$ = " + "{0:.2f},    ".format(self.rs,2) +
				r"$R_{NPA}$ = " + "{0:.2f}\n".format(self.R)  +
        			r"$T_e$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.Te, self.Te*AU_to_eV) +
        			r"$\mu$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.μ, self.μ*AU_to_eV) +
        			r"$\mu-V(R)$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.μ-self.vxc_f(self.ne)[-1],(self.μ-self.vxc_f(self.ne)[-1])*AU_to_eV) +
        			r"$Z^\ast =$ " + "{0:.2f}".format(self.Zstar)  )
			self.nice_text = text
			props = dict(boxstyle='round', facecolor='w')
			ax.text(0.02,0.05, text, fontsize=15, transform=ax.transAxes, verticalalignment='bottom', bbox=props)

		plt.tight_layout()
		name = "NPA_{0}_rs{1}_{2}eV_R{3}.png".format(self.name, np.round(self.rs,2), np.round(self.Te*AU_to_eV,2) ,np.round(self.R))
		plt.savefig(os.path.join(PACKAGE_DIR,'media',name), dpi=300, bbox_inches='tight',facecolor="w")
		if show == True:
			plt.show()
		return fig, axs

	def make_plot_bound_free(self, show=False):

		fig, axs = plt.subplots(ncols=2,figsize=(20,8),facecolor='w')
  	      
		# Density * 4pi r^2 plot
		factor = 4*np.pi*self.grid.xs**2
		# axs[0].plot(self.petrov.r_data, 4*np.pi*self.petrov.r_data**2*(self.petrov.rho_data + self.petrov.rho_0), 'k--', label="Petrov AA")
		axs[0].plot(self.grid.xs, self.ne*factor ,'--.k',label=r'$n_e$')
		axs[0].plot(self.grid.xs, self.nb*factor, label=r'$n_b$')
		axs[0].plot(self.grid.xs, self.nf*factor, label=r'$n_f$')
		
		
		axs[0].set_ylabel(r'$4 \pi r^2 n_e(r) $ [A.U.]',fontsize=20)
		axs[0].set_ylim(1e-2, 1e3)
		axs[0].set_yscale('log')

		# Density ne plot
		# axs[1].plot(self.petrov.r_data, self.petrov.rho_data + self.petrov.rho_0, 'k--', label="Petrov AA")
		axs[1].plot(self.grid.xs, self.ne , 'k', label=r'$n_e$')
		axs[1].plot(self.grid.xs, self.nb, label=r'$n_b$')
		axs[1].plot(self.grid.xs, self.nf, label=r'$n_f$')
		axs[1].plot(self.grid.xs, self.ρi, label=r'$ Z^\ast n^0_i g_{ii}(r) $ ')
		axs[1].plot(self.grid.xs, np.abs(self.ρi - self.ne), label=r'$|\Sigma_j \rho_j|$ ')

		axs[1].set_ylabel(r'$n_e$ [A.U.]',fontsize=20)
		axs[1].set_ylim(self.ρi[-1]*1e-3,1e6)
		#axs[1].set_yscale('symlog',linthresh=1e-3)
		axs[1].set_yscale('log')
  
		for ax in axs:
			ax.set_xlim(self.grid.xs[0],self.grid.xs[-1])
			ax.set_xscale('log')
			ax.set_xlabel(r'$|r-R_1|$ [A.U.]',fontsize=20)
			ax.legend(loc="upper right",fontsize=20,labelspacing = 0.1)
			ax.tick_params(labelsize=20)
			ax.grid(which='both',alpha=0.4)

			# make textbox
			text = ("{0}\n".format(self.name)+ 
				r"$r_s$ = " + "{0},    ".format(np.round(self.rs,2)) +
				r"$R_{NPA}$ = " + "{0}\n".format(self.R)  +
        			r"$Te$ = " + "{0} [A.U.] = {1} eV\n".format(np.round(self.Te,2),np.round(self.Te*AU_to_eV,2)) + r"$\mu$ = " + "{0} [A.U.]\n".format(np.round(self.μ,2)) +
        			r"$Z^\ast = $" + "{0}".format(np.round(self.Zstar,2))  )

			props = dict(boxstyle='round', facecolor='w')
			ax.text(0.05,0.95, text, fontsize=15, transform=ax.transAxes, verticalalignment='top', bbox=props)

		plt.tight_layout()
		name = "NPA_densities_{0}_rs{1}_{2}eV_R{3}.png".format(self.name, np.round(self.rs,2), np.round(self.Te*AU_to_eV,2) ,np.round(self.R))
		plt.savefig(os.path.join(PACKAGE_DIR,'media',name), dpi=300, bbox_inches='tight',facecolor="w")
		if show == True:
			plt.show()
		return fig, axs

	def make_charge_plot(self, show=False):
		fig, axs = plt.subplots(ncols=2,figsize=(20,8),facecolor='w')
  	      
		# Density * 4pi r^2 plot
		factor = 4*np.pi*self.grid.xs**2
		for ax in axs:
			# ax.plot(self.petrov.r_data, 4*np.pi*self.petrov.r_data**2*(self.petrov.rho_data + self.petrov.rho_0), 'k--', label="Petrov AA")
			ax.plot(self.grid.xs, self.ne*factor ,'--.k',label=r'$n_e$')
			ax.plot(self.grid.xs, self.nb*factor, label=r'$n_b$')
			ax.plot(self.grid.xs, self.nf*factor, label=r'$n_f$')

			ax.set_ylabel(r'$4 \pi r^2 n_e(r) $ [A.U.]',fontsize=20)
			ax.set_xlabel(r'$|r-R_1|$ [A.U.]',fontsize=20)
			ax.legend(loc="upper right",fontsize=20,labelspacing = 0.1)
			ax.tick_params(labelsize=20)
			ax.grid(which='both',alpha=0.4)

		# axs[0].set_ylim(1e-2, np.max(factor*self.ne*1.5))
		axs[0].set_ylim(0, np.max((factor*self.ne)[:self.rws_index]*1.5))
		# axs[0].set_xlim(0, self.grid.xs[np.argmax(factor*self.nb)]*2)
		axs[0].set_xlim(0, self.rs)
		
		axs[1].set_ylim(1e-2, np.max(factor*self.ne*1.5))
		axs[1].set_xlim(0, self.grid.xs[-1])
		

		# make textbox
		text = ("{0}\n".format(self.name)+ 
			r"$r_s$ = " + "{0},    ".format(np.round(self.rs,2)) +
			r"$R_{NPA}$ = " + "{0}\n".format(self.R)  +
  			r"$Te$ = " + "{0} [A.U.] = {1} eV\n".format(np.round(self.Te,2),np.round(self.Te*AU_to_eV,2)) + r"$\mu$ = " + "{0} [A.U.]\n".format(np.round(self.μ,2)) +
  			r"$Z^\ast = $" + "{0}".format(np.round(self.Zstar,2))  )

		props = dict(boxstyle='round', facecolor='w')
		axs[0].text(0.05,0.95, text, fontsize=15, transform=axs[0].transAxes, verticalalignment='top', bbox=props)

		plt.tight_layout()
		name = "NPA_densities_{0}_rs{1}_{2}eV_R{3}.png".format(self.name, np.round(self.rs,2), np.round(self.Te*AU_to_eV,2) ,np.round(self.R))
		plt.savefig(os.path.join(PACKAGE_DIR,'media',name), dpi=300, bbox_inches='tight',facecolor="w")
		if show == True:
			plt.show()
		return fig, axs

if __name__=='__main__':
	# 				 Z    A   T[AU] rs  R
	# atom = NeutralPseudoAtom(13, 28, 1050*Kelvin,  3.1268 , 10, Npoints=1000,name='Aluminum', TFW=False, ignore_vxc=True)
	atom = NeutralPseudoAtom(13, 28, 1*eV,  3 , 30, Npoints=3000, Zstar_init=3.8, name='Aluminum', TFW=False, ignore_vxc=False)

	# folder= "os.path.join(PACKAGE_DIR, 'data')"
	# fname = "Aluminum_NPA_TFD_R9.0e+00_rs3.0e+00_T3.7e-02eV_Zstar3.8.dat"
	# atom  = load_NPA(folder + fname,TFW=False, ignore_vxc=False)
	#atom.solve_TF(verbose=True)
	atom.solve_NPA(verbose=True)
	
	# atom.solve_TF(verbose=True)
	atom.save_data()
	atom.make_plots()
	
	# print(atom.TF.n_TF(1,1e-20))