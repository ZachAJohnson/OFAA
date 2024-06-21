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

from .grids import FourierGrid
from .solvers import jacobi_relaxation, sor, gmres_ilu, tridiagsolve
from .physics import FermiDirac, ThomasFermi, χ_Lindhard, χ_TF, More_TF_Zbar, Fermi_Energy, n_from_rs, Debye_length

import matplotlib.pyplot as plt
from matplotlib import colors

import json

import pandas as pd

# def load_NPA( fname, TFW=True, ignore_vxc=False):
# 	#e.g. fname="/home/zach/plasma/atomic_forces/average_atom/data/NPA_Aluminum_TFD_R9.0e+00_rs3.0e+00_T3.7e-02eV_Zstar3.0.dat"
# 	with open(fname) as f:
# 	    line = f.readlines()[1]
# 	info_dict = json.loads(line.strip("\n"))
# 	name  = info_dict['name']
# 	μ     = info_dict['μ[AU]']
# 	Z     = info_dict['Z']
# 	A     = info_dict['A']
# 	Zstar = info_dict['Zstar']
# 	T     = info_dict['T[AU]']
# 	rs    = info_dict['rs[AU]']


# 	data = pd.read_csv(fname, delim_whitespace=True, header=1, comment='#')
# 	ne   = data['n[AU]']
# 	n_f  = data['nf[AU]']
# 	n_b  = data['nb[AU]']
# 	ni   = data['n_ion[AU]']
# 	φe   = data['(φ_e+φ_ions)[AU]']
# 	φion = data['φtotal[AU]']-φe

# 	xs = data['r[AU]']
# 	R  = np.array(xs)[-1]
# 	N  = len(xs)

# 	NPA = NeutralPseudoAtom(Z, A, T, rs, R, name=name, initialize=False, TFW=TFW, Npoints=N, ignore_vxc=ignore_vxc)
# 	NPA.μ    = float(μ)
# 	NPA.ne   = np.array(ne)
# 	NPA.n_b  = np.array(n_b)
# 	NPA.n_f  = np.array(n_f)
# 	NPA.φe   = np.array(φe)
# 	NPA.φion = np.array(φion)
# 	NPA.Zstar= float(Zstar)

# 	NPA.set_physical_params()

# 	NPA.ni   = np.array(ni)
# 	NPA.gii  = NPA.ni / NPA.ni_bar
# 	NPA.make_ρi()

# 	return NPA

class Atom():
	"""
	Basics of an Atom 
	"""
	def __init__(self, Z, A, name=''):
		self.A = A
		self.Z = Z
		self.name = name	

class NeutralPseudoAtom(Atom):
	"""
	A NeutralPseudoAtom class
	"""
	def __init__(self, Z, A, Ti, Te, rs, R, initialize=True, gradient_correction=None, Weizsacker_λ = 1, μ_init = None, Zstar_init = 'More', 
		rmin=2e-2, Npoints=100, iet_R_over_rs = None, iet_N_bins = 2000, name='',ignore_vxc=False, fixed_Zstar = False, use_full_ne_for_nf=False,
		χ_type = 'Lindhard', gii_init_type = 'step', grid_spacing='quadratic'):
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

		# Whether to include xc, mainly because libxc is an absolute pain
		self.TF = ThomasFermi(self.Te, ignore_vxc = ignore_vxc )
		self.μ_init = μ_init
		
		if Zstar_init =='More':
			self.Zstar_init = More_TF_Zbar(self.Z, n_from_rs(rs), self.Te )
			print(f"Using More TF fit for initial Zstar = {self.Zstar_init:0.3f}")
		else:
			self.Zstar_init = Zstar_init

		self.fixed_Zstar = fixed_Zstar # If True, use Zstar_init always, don't update
		self.use_full_ne_for_nf = use_full_ne_for_nf
		self.χ_type = χ_type

		# Instantiate 1-D grid and ThomasFermi
		print("	Intializing grid")
		self.grid = FourierGrid(rmin, R, Npoints)
		self.rws_index = np.argmin(np.abs(self.grid.xs - self.rs))
		self.make_fast_n_TF()

		# sparse grid for vxc and vectorize
		self.ignore_vxc = ignore_vxc
		self.make_vxc_funcs()
		
		# Gradient Corrections
		self.gradient_correction = gradient_correction
		if gradient_correction is not None:
			self.set_gradient_correction()
			self.λ_W = Weizsacker_λ
		
		if initialize:
			# Initializing densities and potentials
			self.reinitialize()
			self.gii_initial = self.gii.copy()

		if rs==R:
			self.aa_type='AA_TF'
		if rs<R:
			self.aa_type='NPA_TF'
		if not self.ignore_vxc:
			self.aa_type += 'D'
		if gradient_correction is not None:
			self.aa_type += gradient_correction

	def get_vxc(self, ne):
		sparse_ne = np.e**interp1d(np.log(self.grid.xs), np.log(ne), kind='cubic', bounds_error=False, fill_value='extrapolate')(np.log(self.sparse_xs))
		vxc_sparse = self.vxc_f(sparse_ne)

		vxc = interp1d(self.sparse_xs, vxc_sparse, kind='linear', bounds_error=False, fill_value='extrapolate')(self.grid.xs)
		return vxc
	

	def make_vxc_funcs(self):
		self.sparse_xs = np.geomspace(np.min(self.grid.xs), np.max(self.grid.xs), num = 200)
		
		if self.ignore_vxc == False:
			self.vxc_f = lambda rho: self.TF.simple_vxc(rho)
			self.vxc_f = np.vectorize(self.vxc_f)
		else:
			self.vxc_f = lambda rho: 0
			self.vxc_f = np.vectorize(self.vxc_f)

	def make_fast_n_TF(self):
		# IMPLEMENT: check if file exists, if not create it	
		etas = np.sort(np.concatenate([np.geomspace(1e-4,1e8,num=10000),-np.geomspace(1e-4,10**(2.5),num=1000)]))
		I12_values = FermiDirac.Ionehalf(etas)
		Ionehalf = interp1d(etas, I12_values, kind='linear', bounds_error=False, fill_value=(0, None))
		self.fast_n_TF = lambda eta: (np.sqrt(2)/np.pi**2)*self.Te**(3/2)*Ionehalf(eta)
		# self.fast_n_TF = np.vectorize(fast_n_TF)

	### Saving data
	def save_data(self):
		header = ("# All units in Hartree [AU] if not specified\n"+
          '{{"name":"{0}", "Z":{1}, "Zstar":{2}, "A":{3},  "μ[AU]": {4:.10e}, "Te[AU]": {5:.3e}, "Ti[AU]": {5:.3e}, "rs[AU]": {6:.3e} }}\n'.format(self.name, self.Z, self.Zstar, self.A, self.μ, self.Te, self.rs)) 
		column_names = f"   {'r[AU]':15} {'n[AU]':15} {'nf[AU]':15} {'nb[AU]':15} {'n_ion[AU]':15} {'φtot[AU]':15} {'δVxc/δρ[Au]':15} {'U_ei[AU]':15} {'U_ii[AU]':15} {'g_ii':15} "
		header += column_names
		data = np.array([self.grid.xs, self.ne, self.n_f, self.n_b, self.ni, self.φe + self.φion, self.get_vxc(self.ne), self.Uei, self.uii_eff, self.gii_from_iet] ).T
		
		txt='{0}_{1}_R{2:.1e}_rs{3:.1e}_T{4:.1e}eV_Zstar{5:.1f}.dat'.format(self.name, self.aa_type, self.R, self.rs, self.Te*AU_to_eV, self.Zstar)
		self.savefile = os.path.join(PACKAGE_DIR,"data",txt)
		np.savetxt(self.savefile, data, delimiter = ' ', header=header, fmt='%15.6e', comments='')
		
	def set_physical_params(self):
		self.ni_bar = 1/self.WSvol # Average Ion Density of plasma
		self.ne_bar = self.Zstar * self.ni_bar # Average Electron Density of plasma
	
		self.EF  = Fermi_Energy(self.ne_bar)
		self.kF = (2*self.EF)**(1/2)

		self.λTF = Debye_length(self.Te, self.ni_bar, self.Zstar)
		self.kTF = 1/self.λTF

		self.κ   = self.λTF*self.rs
		self.φ_κ_screen = self.κ
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

		print("Intialized Potentials and Densities")

	### Initializing
	def initialize_ne(self):
		"""
		Initial Guess for electron charge densit using Debye-Huckel exponent
		"""
		
		# Use approximate density based on plasma electron density, ignoring unknown W correction
		self.ne = self.ne_bar * np.ones_like(self.grid.xs)
		# if self.rs == self.R:
		self.set_μ_infinite()
		# else:
		# 	self.set_μ_infinite()

		eta_approx = self.μ/self.Te - self.get_βVeff(self.φe, self.ne, self.ne_bar)

		self.ne_init_core  = self.fast_n_TF( eta_approx ) #self.Z/self.WSvol*self.grid.ones
		if self.rs == self.R:
			self.ne_init = self.ne_init_core
			self.ne = self.ne_init.copy() # Update actual density
		else:
			transition_func = np.exp(-0.5*(self.grid.xs/(0.5*self.rs))**2)
			self.ne_init_outer = (1-transition_func)*self.ρi*np.ones_like(self.grid.xs)
			netQ_outer = self.grid.integrate_f(self.ρi - self.ne_init_outer)
			remaining_Q = self.Z + netQ_outer 
			self.ne_init_core *= transition_func
			self.ne_init_core *= remaining_Q/self.grid.integrate_f(self.ne_init_core)
			self.ne_init = self.ne_init_core + self.ne_init_outer
			self.ne = self.ne_init.copy() # Update actual density
			self.set_μ_infinite()

		
		self.n_b, self.n_f = self.grid.zeros.copy(), self.grid.zeros.copy() #Initializing bound, free 

	def get_βVeff(self, φe, ne, ne_bar):
		if self.ignore_vxc and self.gradient_correction is None:
			βVeff = ( -φe - self.φion )/self.Te
		elif self.ignore_vxc==False and self.gradient_correction is None:
			βVeff = ( -φe - self.φion + self.get_vxc(ne) - self.vxc_f(ne_bar) )/self.Te
		elif self.ignore_vxc and self.gradient_correction is not None:
			βVeff = ( -φe - self.φion + self.get_gradient_energy(ne))/self.Te
		elif self.ignore_vxc==False and self.gradient_correction is not None:
			βVeff = ( -φe - self.φion + self.get_gradient_energy(ne) + self.get_vxc(ne) - self.vxc_f(ne_bar))/self.Te

		return βVeff

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

	def set_gradient_correction(self):
		"""
		Sets the coefficients and overall gradient correction energy
		[1] - Mod-MD Murillo paper <https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021044>
		Args:
			str gradient_correction_type: 'K' is Kirzhnits, 'W' is Weizsacker for arbitrary λ

		"""
		if self.gradient_correction=='K':
			K_coeff_func = lambda *args: self.TF.K_correction_coefficients(*args, second_b_term_zero=1)
			self.get_grad_coeffs = np.vectorize(K_coeff_func)
		elif self.gradient_correction=='W':
			self.get_grad_coeffs = lambda *args: [ -1/4*self.λ_W, 1/8*self.λ_W ]
			self.get_grad_coeffs = np.vectorize(self.get_grad_coeffs)

	def get_gradient_energy(self, ne):
		"""
		Defined coefficients by
		δF_K/δρ = a/ρ nabla^2 (ρ) + b/ρ^2 |nabla(ρ)|^2 
		"""
		ne_k = self.grid.FT_r_2_k(ne)
		grad_n_square = self.grid.FT_k_2_r(self.grid.ks*ne_k)
		laplace_ne = -self.grid.FT_k_2_r(self.grid.ks**2*ne_k)

		if self.gradient_correction=='K':
			eta = self.TF.η_interp(ne)
			aW, bW = self.get_grad_coeffs(eta, ne, self.T)
		elif self.gradient_correction=='W':
			aW, bW = self.get_grad_coeffs()
		return aW/ne * laplace_ne + bW/ne**2 * grad_n_square
	

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
			self.χee = lambda k: χ_Lindhard(k, self.kF)
			self.χee = np.vectorize(self.χee)

	def make_Uei(self):
		if self.use_full_ne_for_nf == True:
			nf = self.ne.copy()
			nf = np.where(nf<=1e-30, 1e-30, nf)
		else:
			nf = self.n_f.copy() #+ self.ni_bar
			nf = np.where(nf<=1e-30, 1e-30, nf)

		etas = self.TF.η_interp(nf) # η = β( μ + self.φe + self.φion - self.get_vxc(ne)  )         
		if self.ignore_vxc:
			totφ_pseudo = etas*self.Te - self.μ # total potential that must be acting on nf
		else:
			totφ_pseudo = etas*self.Te + self.get_vxc(nf) - self.vxc_f(self.ne_bar)  - self.μ # total potential that must be acting on nf
		φe_from_nf, _ = self.get_φe( (-nf + self.ρi)  ) # potential from nf itself

		φ_pseudo = (totφ_pseudo - φe_from_nf)
		self.Uei = φ_pseudo + self.Zstar/self.R

	def make_Uei_iet(self):
		Uei_iet_interp = interp1d(self.grid.xs, self.Uei , bounds_error=False, fill_value='extrapolate')
		
		@np.vectorize
		def get_fixed_Uei(r):
		    if r<self.grid.xs[0]:
		        return self.Uei[0]
		    elif r>self.grid.xs[-1]:
		        return self.Zstar/r
		    else:
		        return Uei_iet_interp(r)

		self.Uei_iet = get_fixed_Uei(self.iet.r_array*self.rs)
		self.Uei_iet_k = self.rs**3 * self.iet.FT_r_2_k( self.Uei_iet)


	def set_uii_eff(self):
		self.set_physical_params()
		self.make_Uei()
		self.make_Uei_iet()
		
		u_k_Y_approx = 4*π*self.Zstar**2/(  (self.iet.k_array/self.rs)**2 + self.κ**2)
		u_r_Y_approx = self.Zstar**2/(self.iet.r_array*self.rs)*np.exp(-self.κ*self.iet.r_array)

		self.uii_k_eff_iet = 4*π*self.Zstar**2/(self.iet.k_array/self.rs)**2 + self.χee(self.iet.k_array/self.rs)*self.Uei_iet_k**2 - u_k_Y_approx
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
			self.gii = self.gii_from_iet.copy()
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
		# numerical rest of charge
		self.ρ_r = ρ.copy()
		self.ρ_k = self.grid.FT_r_2_k(self.ρ_r)
		self.φe_k = 4*π*self.ρ_k/self.grid.ks**2
		self.φe_r = self.grid.FT_k_2_r(self.φe_k)
		φe = self.φe_r.copy()
		φe -= φe[-1]
		return φe, np.zeros_like(φe)

	def get_φe(self, ρ):
		# numerical rest of charge
		κ = 1/self.rs # self.kTF
		self.ρ_r = ρ.copy()
		self.ρ_k = self.grid.FT_r_2_k(self.ρ_r)
		if False:#self.rs != self.R:
			n_k_screen = κ**2/(4*π) * self.grid.FT_r_2_k(self.φe)
			self.φe_k = 4*π*(self.ρ_k + n_k_screen)/(self.grid.ks**2 + κ**2)
		else:
			self.φe_k = 4*π*self.ρ_k/self.grid.ks**2
		self.φe_r = self.grid.FT_k_2_r(self.φe_k)
		φe = self.φe_r.copy()
		φe -= φe[-1]
		return φe, np.zeros_like(φe)
	
	# def get_φe(self, ρ):
	# 	# Setting up analytic ρ, φ for exact cancellation with central ion
	# 	## iet space
	# 	r, Z, rs, R = self.iet.r_array*self.rs, self.Z, self.rs, self.R
	# 	λ=20/rs # rs
	# 	Q = self.grid.integrate_f(ρ) # -Z
	# 	self.iet.ρ0_r = λ**3*Q/(8*π) * np.exp(-λ*r) # analytically normalized to cancel Z

	# 	### aa space
	# 	r, Z, rs, R = self.grid.xs, self.Z, self.rs, self.R
	# 	ε = 1e-10 # ε->0
	# 	Γ = gammaincc(ε, r*λ)*gamma(ε) 
	# 	self.φe0_iet = Q/r*(1-np.exp(-r*λ)) - 1/2*Q*λ**2*Γ
	# 	self.ρ0_r_iet = λ**3*Q/(8*π) * np.exp(-λ*r)

	# 	# numerical rest of charge
	# 	self.iet.ρ_r = interp1d(self.grid.xs, ρ, bounds_error=False, fill_value='extrapolate')(self.iet.r_array*self.rs)
	# 	self.iet.δρ_r = self.iet.ρ_r - self.iet.ρ0_r
	# 	self.iet.δρ_k = self.iet.FT_r_2_k(self.iet.δρ_r)
	# 	self.iet.δφe_k = 4*π*self.iet.δρ_k/(self.iet.k_array/self.rs)**2
	# 	self.iet.δφe_r = self.iet.FT_k_2_r(self.iet.δφe_k)

	# 	self.iet.δφe_r = self.iet.δφe_r
	# 	δφe = interp1d(self.iet.r_array*self.rs, self.iet.δφe_r, bounds_error=False, fill_value='extrapolate')(self.grid.xs)
	# 	φe = self.φe0_iet + δφe
	# 	φe -= φe[-1]
	# 	return φe, np.zeros_like(φe)

	# def update_φe(self, l_decay = None):
	# 	if l_decay is None:
	# 			l_decay = 1/(0.25*self.kTF)
		
	# 	φe_guess, rel_errs = self.get_φe(self.ρi - self.ne)

	# 	short_distance_weight = np.exp(-self.grid.xs/l_decay)
	# 	short_distance_weight = 10**(-5*self.grid.xs/self.R)
	# 	self.φe = (1-short_distance_weight)*self.φe + short_distance_weight*φe_guess

	# 	return np.mean(rel_errs)

	# def get_ne_W(self, constant_hλ=False):

	# 	def banded_Ab():
	# 		d2dx2 = self.grid.matrix_d2fdx2()

	# 		etas = self.TF.η_interp(self.ne)
	# 		if self.gradient_correction=='K':
	# 			aW, bW = self.get_grad_coeffs(etas, self.ne, self.Te)
	# 			print("bW term NOT supported, please switch to 'W' gradient_correction_type for now.")
	# 		elif self.gradient_correction=='W':
	# 			aW, bW = self.get_grad_coeffs()

	# 		if not self.ignore_vxc:
	# 			bc_vec  = -1/(2*aW)*(self.Te*etas - (self.μ + self.ϕe + self.ϕion - self.get_vxc(self.ne) + self.vxc_f(self.ne_bar)) )
	# 		if self.ignore_vxc:
	# 			bc_vec  = -1/(2*aW)*(self.Te*etas - (self.μ + self.ϕe + self.ϕion ))
					
	# 		# Split to gaurantee positive solution 
	# 		γ = np.sqrt(self.ne)*self.grid.xs
	# 		b = np.max([0*self.grid.xs, -bc_vec], axis=0)*γ # will go on right side
	# 		c = np.max([0*self.grid.xs,  bc_vec], axis=0) # will go on left side

	# 		A = -d2dx2 + np.diag(c)

	# 		#Boundary
	# 		dx= self.grid.dx
	# 		x = self.grid.xs
			
	# 		# Interior- Kato cusp condition
	# 		interior_bc = 'kato'
	# 		if interior_bc in ['kato','Kato']:
	# 			# A[0,0]  = 1; A[0,1] = -1/(1 + self.grid.dx[0]*(-self.Z+ 1/self.grid.xs[1])) 
	# 			# A[0,0:2] *= 1/self.grid.dx[0]**2  #get number more similar to other matrix values
	# 			# b[0] = -(self.Z + 1/self.grid.xs[0])*γ[0]/self.grid.dx[0]
				
	# 			# retry- use Z~0 for rmin<<1, so exactly γ(r)=r/r_0 γ(r_0), applied to first point, or γ[1] - γ[0]x[1]/x[0]=0
	# 			A[0,0] = -x[1]/x[0]
	# 			A[0,1] = 1
	# 			b[0]   = 0
				

	# 		elif interior_bc == 'zero': # Sets γ to zero
	# 			A[0,0] = 1/self.grid.dx[0]**2
	# 			b[0] = 0

	# 		# Exterior- γ''= 0 at edge is nice
	# 		# A[-1,-1] =  2/dx[-1]  /(dx[-1] + dx[-2])
	# 		# A[-1,-2] = -2/(dx[-1]*dx[-2])           
	# 		# A[-1,-3] =  2/dx[-2]/(dx[-1] + dx[-2])
			
	# 		# Exterior- dγ=γ/r
	# 		A[-1,-1] = 1/dx[-1]**2 
	# 		A[-1,-2] = -1/dx[-1]**2 

	# 		b[-1]  = γ[-1]/dx[-1]/x[-1] 
			

	# 		return A, b		

	# 	A, b = banded_Ab()
	# 	self.Ab_W = A, b
	# 	γs = tridiagsolve(A, b)
	# 	self.γs = γs

	# 	new_ne = γs**2/self.grid.xs**2
	# 	new_ne = new_ne * self.Z/self.grid.integrate_f(new_ne) # normalize so it is reasonable

	# 	return new_ne

	def get_new_ne(self, **kwargs):
		if self.gradient_correction is not None:
			new_ne = self.get_new_ne_W(**kwargs)
		else:
			new_ne = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)

	def update_ne(self, alpha, **kwargs):
		if False:#self.gradient_correction is not None:
			# Above updates very slowly when gradients are very small, so we add in the update based on TF integral  	
			ne_guess = self.get_ne_W(alpha )
		else:
			ne_guess = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)

		self.ne = (1-alpha)*self.ne + alpha*ne_guess

	def small_update(self, f_new, f_old, alpha, type = 'exp', l_decay = None):
		delta = f_new - f_old	

		small_steps = np.array([alpha*delta , 0.1*np.sign(delta)*f_old])

		#Update relative to size of current rho
		if type=='rel':
			f_return = f_old + small_steps[0]

		#Update absolute 
		elif type=='abs':
			f_return = f_old + small_steps[1]

		# Minimum of two
		elif type=='both':
			absmin = lambda vec_list: np.array([  vec[ np.argmin(np.abs(vec))] for vec in vec_list])

			smallest_steps = absmin(small_steps.T)
		
			f_return = f_old + smallest_steps

		elif type=='exp':
			absmin = lambda vec_list: np.array([  vec[ np.argmin(np.abs(vec))] for vec in vec_list])

			smallest_steps = absmin(small_steps.T)
			
			if l_decay is None:
				l_decay = 1/(0.25*self.kTF)
			short_distance_weight = np.exp(-self.grid.xs/l_decay)
			# short_distance_weight = np.exp(-np.log(10)* 4*self.grid.xs/self.grid.xmax ) 

			f_return = f_old + smallest_steps*short_distance_weight

		self.steps = small_steps[0]
		return f_return

	def make_bound_free(self, φ_shift):
		etas = self.TF.η_interp(self.ne)
		xmid = -self.get_βVeff(self.φe, self.ne, self.ne_bar)
		c = 0.05
		fcut = (1 + np.exp(-1/c))/( 1 + np.exp( ( self.grid.xs - self.rs)/(c*self.rs)) )

		self.n_b = ThomasFermi.n_bound_TF(self.Te, etas, xmid )*fcut
		self.n_f = self.ne - self.n_b  
		
	def update_bound_free(self, φ_shift=0, alpha = 2e-1):
		"""
		Gets bound free separation using approximation  in ThomasFermi. 
		Only iteratively updates Zstar
		Returns: 
			Exact Zstar using bound, free.

		"""
		self.make_bound_free(φ_shift)

		self.new_Zstar_guess = self.Z - self.grid.integrate_f(self.n_b)

		try:
			ΔZ_new = (self.new_Zstar_guess - self.Zstar)
			ΔZ_old = (self.old_Zstar_guess - self.old_Zstar)
			dZguess_dZ = (ΔZ_new - ΔZ_old)/(self.Zstar - self.old_Zstar)
			self.old_Zstar = self.Zstar
			self.Zstar = self.Zstar - ΔZ_new/dZguess_dZ
		except AttributeError:	
			self.old_Zstar = self.Zstar
			self.Zstar = self.Zstar + alpha*(self.new_Zstar_guess-self.Zstar) #smaller nonzero update
		
		self.old_Zstar_guess = self.new_Zstar_guess

		if self.Zstar<=0.01:
			self.Zstar = 0.01


		#Update ne_bar, free density etc.
		self.set_physical_params() 
		self.make_ρi()

	def compensate_ρe(self):#, delta_Q):
		"""
		After Zstar updates, Q changes by alot. To get better initial guess for this system,
		We increase the electron density to remain overall neutrality and get better numerics.
		"""
		# delta_ne = self.get_Q()/self.Vol #delta_Q/self.Vol 
		# self.ne += delta_ne  #Increase by delta_Q overall 
		# Qe = -(self.get_Q() - self.Qion)
		# self.ne = self.ne*self.Qion/Qe  #Increase by delta_Q overall 

		# short_distance_weight = np.exp(-0.5*self.kTF*self.grid.xs)
		# self.ne = self.ne*() 


		# self.n_f = self.n_f*self.Qion/Qe  #Increase by delta_Q overall 
		# self.n_b = self.n_b*self.Qion/Qe  #Increase by delta_Q overall 

	def update_ρi_and_Zstar_to_make_neutral(self, alpha = 1):
		Zstar_needed_for_neutral = (self.grid.integrate_f(self.ne)-self.Z)/self.grid.integrate_f(self.ni)
		self.Zstar = self.Zstar + alpha*(Zstar_needed_for_neutral - self.Zstar) #smaller nonzero update
		if self.Zstar<=0.01:
			self.Zstar = 0.01

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
	
	def solve_TF(self, verbose=False, picard_alpha = 1e-2, nmax = 1e4, tol=1e-4, remove_ion = False, save_steps=False, n_wait_update_Zstar = 100):
		"""
		Solve TF OFDFT equation, assuming a given Zbar for the plasma ions 
		"""
		if verbose:
			print("Beginning self-consistent electron solver.")
			print("_________________________________")
		if remove_ion == True:
			self.φion = 0*self.grid.xs
			self.Qion = self.grid.integrate_f( self.ρi )

		self.ne_list = [self.ne.copy()]
		self.ρi_list = [self.ρi.copy()]
		self.ne_bar_list = [self.ne_bar]
		self.μ_list, self.rho_err_list, self.change_list = [self.μ], [0], [0]
		self.φe_list = [self.φe]
		
		Q = 0
		n = 0
		converged, μ_converged, Zbar_converged = False, False, False
		while not converged and n < int(nmax):
			Q_old = Q
			old = self.μ, np.mean(self.ne), np.mean(self.φe) 
			
			# Update physics in this order
			self.φe, poisson_err = self.get_φe(self.ρi - self.ne)
			poisson_err = np.mean(poisson_err)
			self.update_ne(picard_alpha)
			self.new_Zstar_guess = self.Zstar
			if remove_ion: #Simulating removal of ion, keep μ the same.
				pass
			else:
				if self.rs==self.R:
					# get Zstar from bound/free
					if self.fixed_Zstar == False:
						self.update_bound_free()

					if n%10==0 or n<5 and not μ_converged:
						self.set_μ_neutral()
					elif not μ_converged: 
						self.update_μ_newton(alpha1=1e-3)
				else:
					# if μ_converged==True and Zbar_converged==False and n>n_wait_update_Zstar :
					# 	old_ne_bar = self.ne_bar
					# 	if self.fixed_Zstar == False:
					# 		new_Zstar = self.update_bound_free(alpha=1e-1 ) # also picard updates Zstar
					# 		if np.abs(new_Zstar/self.Zstar - 1)<1e-5:
					# 			Zbar_converged = True
					# 		self.ne += self.ne_bar - old_ne_bar
					# if self.fixed_Zstar == False:
					# 	self.update_ρi_and_Zstar_to_make_neutral() 
					# self.set_μ_infinite()

					
					if self.fixed_Zstar == False and n>n_wait_update_Zstar:
						if n%10==0:
							old_ne_bar = self.ne_bar
							self.update_bound_free(alpha=1e-1 ) # also picard updates Zstar
							self.ne   += self.ne_bar - old_ne_bar
					
					self.update_ρi_and_Zstar_to_make_neutral()
					# self.set_μ_infinite()
					self.set_μ_neutral()

					
					
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

			if np.abs(1-self.μ/old[0])<1e-3:
				μ_converged = True

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
			if remove_ion:
				if  change<tol and abs(rho_err)<tol:
					converged=True
			else:
				if abs(Q)<1e-3 and change<tol and abs(rho_err)<tol:
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
		return converged

	def update_Zstar(self, new_Zstar):
		# Bisection algorithm to bound the function f_Z
		# Update with ITP method next
		f_Z = new_Zstar - self.Zstar

		if   new_Zstar < self.Zstar:
				self.max_Zstar = self.Zstar
				# self.min_Zstar = np.max([new_Zstar, self.min_Zstar])
		elif new_Zstar > self.Zstar:
				self.min_Zstar = self.Zstar
				# self.max_Zstar = np.min([new_Zstar, self.max_Zstar])

		self.Zstar = (self.min_Zstar + self.max_Zstar)/2

	def run_empty_TF(self, **kwargs):
		# Save current info and run empty-ion shell 
		old_ne   = self.ne.copy()   
		old_φe   = self.φe.copy()   
		old_φion = self.φion.copy() 
		old_μ    = self.μ.copy()

		# Runs TF to get empty-atom electron density
		self.ne = self.ne_bar*np.ones(self.grid.Nx)
		self.solve_TF(remove_ion = True, **kwargs)
		self.empty_ne = self.ne.copy()

		print("Empty ion ran and got μ = {0:.3e} from initial μ = {1:.3e}".format(self.μ, old_μ))
		# Reset actual information to old stuff
		self.ne = old_ne
		self.φe = old_φe
		self.φion = old_φion
		self.μ    = old_μ
		self.make_ρi() #reset Qion 

		# Result - Compute fluctuation density
		self.δn_f = self.n_f - self.empty_ne  # Linear response free density


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
    			r"$\mu-V(R)$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.μ-self.get_vxc(self.ne)[-1],(self.μ-self.get_vxc(self.ne)[-1])*AU_to_eV) +
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
		if not self.ignore_vxc:
			axs[0].plot(self.grid.xs , -self.get_vxc(self.ne) , label=r"$-v_{xc}[n_e]$")
		if self.gradient_correction is not None:
			axs[0].plot(self.grid.xs , -self.get_gradient_energy(self.ne) , label=r"$-v_{W}[n_e]$")
		axs[0].plot(self.grid.xs , -self.get_βVeff(self.φe, self.ne, self.ne_bar)*self.Te , label=r"$-V_{\rm eff}$")


		axs[0].set_ylabel(r'$\phi$ [A.U.]',fontsize=20)
		axs[0].set_ylim(-1e3,1e6)
		axs[0].set_yscale('symlog',linthresh=1e-8)

		# Density ne plot
		# axs[1].plot(self.petrov.r_data, self.petrov.rho_data + self.petrov.rho_0, 'k--', label="Petrov AA")
		axs[1].plot(self.grid.xs, self.ne , label=r'$n_e$')
		axs[1].plot(self.grid.xs, self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar) , label=r'$\frac{\sqrt{2}}{\pi^2}T^{3/2}\mathcal{I}_{1/2}(\eta)$')
		axs[1].plot(self.grid.xs, self.n_b, label=r'$n_b$')
		axs[1].plot(self.grid.xs, self.n_f, label=r'$n_f$')
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
			ax.legend(loc="upper right",fontsize=20,labelspacing = 0.1)
			ax.tick_params(labelsize=20)
			ax.grid(which='both',alpha=0.4)

			# make textbox
			text = ("{0}, {1}\n".format(self.name, self.aa_type.replace("_", " "))+ 
				r"$r_s$ = " + "{0:.2f},    ".format(self.rs,2) +
				r"$R_{NPA}$ = " + "{0:.2f}\n".format(self.R)  +
        			r"$T_e$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.Te, self.Te*AU_to_eV) +
        			r"$\mu$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.μ, self.μ*AU_to_eV) +
        			r"$\mu-V(R)$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.μ-self.get_vxc(self.ne)[-1],(self.μ-self.get_vxc(self.ne)[-1])*AU_to_eV) +
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
		axs[0].plot(self.grid.xs, self.n_b*factor, label=r'$n_b$')
		axs[0].plot(self.grid.xs, self.n_f*factor, label=r'$n_f$')
		
		
		axs[0].set_ylabel(r'$4 \pi r^2 n_e(r) $ [A.U.]',fontsize=20)
		axs[0].set_ylim(1e-2, 1e3)
		axs[0].set_yscale('log')

		# Density ne plot
		# axs[1].plot(self.petrov.r_data, self.petrov.rho_data + self.petrov.rho_0, 'k--', label="Petrov AA")
		axs[1].plot(self.grid.xs, self.ne , 'k', label=r'$n_e$')
		axs[1].plot(self.grid.xs, self.n_b, label=r'$n_b$')
		axs[1].plot(self.grid.xs, self.n_f, label=r'$n_f$')
		axs[1].plot(self.grid.xs, self.ρi, label=r'$ Z^\ast n^0_i g_ii(r) $ ')
		axs[1].plot(self.grid.xs, np.abs(self.ρi - self.ne), label=r'$|\Sigma_j \rho_j|$ ')

		axs[1].set_ylabel(r'$n_e$ [A.U.]',fontsize=20)
		# axs[1].set_ylim(self.ρi[-1]*1e-3,1e6)
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
			ax.plot(self.grid.xs, self.n_b*factor, label=r'$n_b$')
			ax.plot(self.grid.xs, self.n_f*factor, label=r'$n_f$')

			ax.set_ylabel(r'$4 \pi r^2 n_e(r) $ [A.U.]',fontsize=20)
			ax.set_xlabel(r'$|r-R_1|$ [A.U.]',fontsize=20)
			ax.legend(loc="upper right",fontsize=20,labelspacing = 0.1)
			ax.tick_params(labelsize=20)
			ax.grid(which='both',alpha=0.4)

		# axs[0].set_ylim(1e-2, np.max(factor*self.ne*1.5))
		axs[0].set_ylim(0, np.max((factor*self.ne)[:self.rws_index]*1.5))
		# axs[0].set_xlim(0, self.grid.xs[np.argmax(factor*self.n_b)]*2)
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