#Zach Johnson 2022
# Neutral Pseudo Atom code

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
from scipy.sparse.linalg import gmres, LinearOperator, eigs
from scipy.optimize import root, newton_krylov, fsolve

from atomic_forces.GordonKim.python.atoms import Potentials, petrov_atom
from atomic_forces.atomOFDFT.python.physics import FermiDirac, ThomasFermi 

import matplotlib.pyplot as plt
from matplotlib import colors


import pandas as pd
import re
import json

eV = 0.0367512 # So 4 eV = 4 * 0.036.. in natural units
Kelvin = 8.61732814974493e-5*eV #Similarly, 1 Kelvin = 3.16... in natural units 
π = np.pi


class LinearGrid():
	"""	
	Simple 1-D grid class based on np.linspace
	"""
	def __init__(self, xmin, xmax, Nx):
		self.xmin = xmin
		self.xmax = xmax
		self.Nx = Nx

		self.make_linear_grid()

		self.zeros = np.zeros((self.Nx)) #Must be .copy() to work
		self.ones  = np.ones ((self.Nx)) #Must be .copy() to work

	def make_linear_grid(self):
		self.xs = np.linspace(self.xmin, self.xmax, num=self.Nx+1, endpoint=True)[1:]
		self.dx = self.xs[1]-self.xs[0]
		self.bulk_indcs = slice(1,-1)
		self.grid_shape = self.Nx

	def integrate_f(self, f):
		#return np.sum(4*π*f*self.xs**2)*self.dx
		return simps(4*π*f*self.xs**2, x = self.xs)

	def dfdx(self, f):
		"""
		Fourth order derivative of some function. Fills end with zeros.
		df ~ O(dx^5) 
		"""
		ffull = self.zeros.copy()
		#middle
		mid_coeffs = [1/12., -2/3, 0, 2/3, -1/12]
		mid_shifts = [-2, -1, 0, 1, 2]
		ffull[2:-2] = (np.sum([c*np.roll(f,-shift) for c, shift in zip(mid_coeffs, mid_shifts)], axis=0)/self.dx)[2:-2]
		
		#EDges
		ffull[0]  = (-0.25*f[4] + 4/3*f[3] -3*f[2] +4*f[1] - 25/12*f[0])/self.dx
		ffull[1]  = (-0.25*f[5] + 4/3*f[4] -3*f[3] +4*f[2] - 25/12*f[1])/self.dx
		ffull[-1] = (0.25*f[-5] - 4/3*f[-4] +3*f[-3] -4*f[-2] + 25/12*f[-1])/self.dx
		ffull[-2] = (0.25*f[-6] - 4/3*f[-5] +3*f[-4] -4*f[-3] + 25/12*f[-2])/self.dx

		return ffull

	def d2fdx2(self, f):
		"""
		Second order derivative of some function f. Fills end with zeros
		d2f ~ O(dx^3)
		"""
		ffull = self.zeros.copy()
		ffull[1:-1] = (f[2:] + f[:-2] - 2*f[1:-1])/self.dx**2
		ffull[0] = (f[0] - 2*f[1] + f[2])/self.dx**2
		ffull[-1] = (f[-1] - 2*f[-2] + f[-3])/self.dx**2
		return ffull



class Atom():
	"""
	Basics of an Atom 
	"""
	def __init__(self, Z, A, name=''):
		self.A = A
		self.Z = Z
		self.name = name	

def load_NPA( fname, TFW=True, ignore_vxc=False):
	#e.g. fname="/home/zach/plasma/atomic_forces/average_atom/data/NPA_Aluminum_TFD_R9.0e+00_rs3.0e+00_T3.7e-02eV_Zstar3.0.dat"
	with open(fname) as f:
	    line = f.readlines()[1]
	info_dict = json.loads(line.strip("\n"))
	name  = info_dict['name']
	μ     = info_dict['μ[AU]']
	Z     = info_dict['Z']
	A     = info_dict['A']
	Zstar = info_dict['Zstar']
	T     = info_dict['T[AU]']
	rs    = info_dict['rs[AU]']


	data = pd.read_csv(fname, delim_whitespace=True, header=1, comment='#')
	ne   = data['n[AU]']
	n_f  = data['nf[AU]']
	n_b  = data['nb[AU]']
	ni   = data['n_ion[AU]']
	φe   = data['(φ_e+φ_ions)[AU]']
	φion = data['φtotal[AU]']-φe

	xs = data['r[AU]']
	R  = np.array(xs)[-1]
	N  = len(xs)

	NPA = NeutralPseudoAtom(Z, A, T, rs, R, name=name, initialize=False, TFW=TFW, Npoints=N, ignore_vxc=ignore_vxc)
	NPA.μ    = μ
	NPA.ne   = np.array(ne)
	NPA.n_b  = np.array(n_b)
	NPA.n_f  = np.array(n_f)
	NPA.φe   = np.array(φe)
	NPA.φion = np.array(φion)
	NPA.Zstar= Zstar

	NPA.set_physical_params()
	NPA.make_ρi()


	return NPA

class NeutralPseudoAtom(Atom):
	"""
	A NeutralPseudoAtom class
	"""
	def __init__(self, Z, A, T, rs, R, initialize=True, TFW=True, μ_init = None, Zstar_init = 3, Npoints=100, name='',ignore_vxc=False):
		super().__init__(Z, A, name=name)
		self.T = T
		self.rs = rs
		self.R = R
		self.TFW = TFW
		self.WSvol = 4/3*π*self.rs**3
		self.Vol   = 4/3*π*self.R**3
		self.ignore_vxc = ignore_vxc
		self.μ_init = μ_init
		self.Zstar_init = Zstar_init


		# Instantiate linspaced 1-D grid and ThomasFermi
		print("Intializing grid")
		self.grid = LinearGrid(0, R, Npoints)
		self.TF = ThomasFermi(T)
		self.vxc_f = self.TF.fast_vxc()

		# For comparison, get George Petrov AA data
		print("Loading Data for Comparison (from George Petrov)")
		self.petrov = petrov_atom(self.Z,self.A,self.T,self.rs)
		self.petrov.make_densities('/home/zach/plasma/atomic_forces/GordonKim/data/TFDW-eta=1-Te=1.dat')

		# Weizsacker function
		self.get_W_coeffs = np.vectorize(self.TF.W_correction_coefficients)

		if initialize:
			# Initializing densities and potentials
			self.reinitialize()

		self.rws_index = np.argmin(np.abs(self.grid.xs - self.rs))
		if rs==R:
			self.aa_type='AA_TF'
		if rs<R:
			self.aa_type='NPA_TF'
		if not self.ignore_vxc:
			self.aa_type += 'D'
		if self.TFW:
			self.aa_type += 'W'

	def print_metric_units(self):
		aBohr = 5.29177210903e-9 # cm
		print("\n_____________________________________\nPlasma Description in A.U. and metric")
		print("n_ion: {0:10.3e} [ A.U.], {1:10.3e} [1/cc]".format(self.ni_bar, self.ni_bar/aBohr**3))
		print("T: {0:10.3e} [A.U.], {1:10.3e} [eV]".format(self.T, self.T/eV ))

		print("\n")


	### Saving data
	def save_data(self):
		header = ("# All units in [AU] if not specified\n"+
          '{{"name":"{0}", "Z":{1}, "Zstar":{2}, "A":{3},  "μ[AU]": {4:.3e}, "T[AU]": {5:.3e}, "rs[AU]": {6:.3e} }}\n'.format(self.name, self.Z, self.Zstar, self.A, self.μ, self.T, self.rs) + 
		  '\t r[AU]  \t n[AU]  \t nf[AU]  \t nb[AU]     n_ion[AU]    (φ_e+φ_ions)[AU]   φtotal[AU]     δVxc/δρ[Au]  ') 
		data = np.array([self.grid.xs, self.ne, self.n_f, self.n_b, self.ni, self.φe, self.φe + self.φion, self.vxc_f(self.ne)] ).T
		
		txt='{0}_{1}_R{2:.1e}_rs{3:.1e}_T{4:.1e}eV_Zstar{5:.1f}.dat'.format(self.name, self.aa_type, self.R, self.rs, self.T, self.Zstar)
		np.savetxt("/home/zach/plasma/atomic_forces/average_atom/data/" + txt, data, 
		           delimiter = ' ', header=header, fmt='%15.5e', comments='')
	
	def set_physical_params(self):
		self.ni_bar = 1/self.WSvol # Average Ion Density of plasma
		self.ne_bar = self.Zstar * self.ni_bar # Average Electron Density of plasma
	
		self.EF = 0.5*(3*π**2*self.ne_bar)**(2/3)
		self.λTF = np.sqrt(  4*π* self.Z*self.ne_bar  /np.sqrt(self.T**2 + (2/3*self.EF)**2)  )
		

	def reinitialize(self):
		self.Zstar  = self.Zstar_init

		# Initialize Physical Parameters		
		self.set_physical_params()

		# Initialize Ion density
		self.make_ρi()

		# Initialize chemical potential
		if self.μ_init==None:
			self.initialize_μ()
		else: 
			self.μ = self.μ_init

		# Initializing potentials
		self.ϕion = self.Z/self.grid.xs - self.Z/self.grid.xmax #1/r with zero at boundary 
		self.φe_init = self.Z/self.grid.xs*np.exp(-self.λTF*self.grid.xs) - self.φion
		# self.φe_init = self.grid.ones.copy() #-self.φion #self.grid.zeros.copy()
		self.φe = self.φe_init
		

		# Initializing densities 
		self.initialize_ne()

		print("Intialized Potentials and Densities")

	def initialize_μ(self):
		# Set μ such that at R, get asymptotic ne_bar right
		eta_bar = self.TF.η_interp(self.ne_bar)
		μ_init  = eta_bar*self.T + self.vxc_f(self.ne_bar)
		self.μ = μ_init  

	### Initializing
	def initialize_ne(self):
		"""
		Initial Guess for electron charge densit using Debye-Huckel exponent
		"""
		
		# Use approximate density based on plasma electron density, ignoring unknown W correction
		eta_approx = 1/self.T*(self.μ + self.ϕe + self.ϕion - self.vxc_f(self.ne_bar))
		self.ne_init = self.TF.n_TF(self.T, eta_approx ) #self.Z/self.WSvol*self.grid.ones
		# Ne_core_init = self.grid.integrate_f(self.ne_init)

		# # Add constant density so that exactly neutral in correlation sphere of radius R
		# Qneeded = Ne_core_init - self.Qion
		# ne_rest = -Qneeded/self.Vol
		# self.ne_init += ne_rest

		self.ne = self.ne_init # Update actual density
		self.n_b, self.n_f = self.grid.zeros.copy(), self.grid.zeros.copy() #Initializing bound, free 

	def get_eta_from_sum(self, ne, μ):
		if self.ignore_vxc and self.TFW==False:
		    eta= (μ + self.ϕe + self.ϕion )/self.T 
		elif self.ignore_vxc==False and self.TFW==False:
		    eta = (μ + self.ϕe + self.ϕion - self.vxc_f(ne))/self.T
		elif self.ignore_vxc and self.TFW==True:
		    eta = (μ + self.ϕe + self.ϕion - self.get_W(self.ne))/self.T 
		elif self.ignore_vxc==False and self.TFW==True:
		    eta = (μ + self.ϕe + self.ϕion - self.get_W(self.ne) - self.vxc_f(ne))/self.T 

		return eta

	def make_ne_TF(self):
		"""
		Sets e-density using self μ
		"""
		self.ne = self.get_ne_TF(self.ne, self.μ)

	def get_ne_TF(self, ne, μ):
		"""
		Generates electron density self.ne_grid from fermi integral
		Args: 
			float μ: chemical potential
		Returns:
			None
		"""		   
		eta=self.get_eta_from_sum(ne, μ)
		ne = self.TF.n_TF(self.T, eta)
		return ne

	
	def get_Q(self):
		return self.Qion - self.grid.integrate_f(self.ne)

	def get_W(self, ne):
		"""
		Defined coefficients by
		δF_K/δρ = a/ρ nabla^2 (ρ) + b/ρ^2 |nabla(ρ)|^2 
		"""
		eta = self.TF.η_interp(ne)
		aW, bW = self.get_W_coeffs(eta, ne, self.T)
		grad_n = self.grid.dfdx(ne)
		laplace_ne = 1/self.grid.xs**2 * self.grid.dfdx(self.grid.xs**2 * grad_n)


		return aW/ne * laplace_ne + bW/ne**2 * grad_n**2

	## Chemical Potential Methods 
	def set_μ_TF(self):
		"""
		Finds μ through enforcing charge neutrality
		"""
		min_μ = lambda μ: abs(  self.Qion - self.grid.integrate_f( self.get_ne_TF(self.ne, μ) ) )**2

		root_and_info = root(min_μ, 0.4,tol=1e-3)

		self.μ = root_and_info['x'][0]
		#self.μ = self.μ #+ np.min([1e-3*(new_μ-self.μ) , 0.01*np.sign(new_μ-self.μ)])
	
	def update_μ_newton(self, alpha1=1e-3, alpha2= 1e-3):
		"""
		Finds μ through enforcing charge neutrality
		"""
		Qnet = self.get_Q()
		eta = self.get_eta_from_sum(self.ne, self.μ)
		nprime = self.TF.n_TF_prime(self.T,eta)
		# grad_Qsquare = -(1/self.T) * Qnet * self.grid.integrate_f(nprime)
		# change_in_μ = min(np.abs(alpha1 * grad_Qsquare), alpha2)
		# change_in_μ = alpha1* Qnet**2/grad_Qsquare
		# self.μ += - np.sign(grad_Qsquare) * change_in_μ
		# self.μ +=  -alpha1 * Qnet**2/grad_Qsquare
		# alpha = 10*alpha1*np.abs(Qnet)
		self.μ +=  alpha1 * self.T*Qnet/np.sqrt(self.grid.integrate_f(nprime)**2 + 0.1)
		# self.μ +=  +alpha1 * self.T*Qnet/np.sqrt(nprime)
		# print("Partial_mu (Q^2): ", grad_Qsquare)

		
	def shift_μ(self):
		"""
		Set ϕ = ϕion + ϕe to be zero at boundary (arbitrary as long as mu is adjusted).
		"""
		shift = - self.ϕe[-1]  # Shift amount
		self.ϕe = self.ϕe + shift # Adjust ϕe
		self.μ  = self.μ  - shift # Adjust μ, compensating shift so overall energy same
		
	## IONS
	def make_gii(self):
		"""
		Initially just simple step function. Later HNC will be used.
		"""

		self.gii = np.heaviside(self.grid.xs-self.rs , 0)

	def make_ρi(self):
		"""
		Ions other than the central ion. The plasma ions based on gii(r).
		"""
		self.make_gii()

		self.ni = self.ni_bar * self.gii # Number density
		self.ρi = self.ni * self.Zstar   # Charge density
		self.Qion = self.Z  + self.grid.integrate_f( self.ρi )
		# print("Qion = {}".format(self.Qion))


	def update_ϕe(self):
		"""
		Use GMRES to solve Poisson Equation for φe
		Returns:
			float err: Maximum residual magnitude of Ax-b 
		"""
		# Define GMRES operators ###
		def get_b_bulk():
			"""
			For Krylov GMRES optimization, Ax=b. This is b part, for Poisson Eqn.
			Args: None
			Returns: 
				1-D vectorlike b: Same shape
			"""

			b = self.grid.zeros.copy()

			b= self.grid.dx*4*π*  ( -self.ne + self.ρi )
			b_bulk = b[self.grid.bulk_indcs]

			return b_bulk.flatten()

		def Abulk_func( ϕbulk): 
		    """
		    For Krylov GMRES optimization, Ax=b. This is A part, for Poisson Eqn.
		    A(v) simulates A@v matrix multiplication 
		    """
		    ϕbulk = ϕbulk.copy()
		    #Set bulk phi
		    ϕ = np.zeros(self.grid.grid_shape)
		    ϕ[self.grid.bulk_indcs] = ϕbulk
		    #Set bc of phi
		    ϕ[0] = ϕ[1] + 1/3 * self.grid.dx*self.grid.xs[1]*self.ne[1] #Zero E-field from electrons, plasma ions
		    Qnet = self.get_Q() + self.Z
		    # ϕ[-1] = ϕ[-2] + self.grid.dx* Qnet/self.R**2 #cancel E-field from central ion
		    ϕ[-1] = -6/11*(  (-3*ϕ[-2] + 3/2*ϕ[-3] - 1/3*ϕ[-4]) - self.grid.dx* Qnet/self.R**2) #O(h^3) cancel of E-field from net charge
		    #Shift
		    ϕ = ϕ - ϕ[-1] 

		    Aϕbulk = self.grid.zeros.copy()[self.grid.bulk_indcs]
		    dφ2dx2 = self.grid.d2fdx2(φ)
		    dφdx   = self.grid.dfdx(φ)
		    Aϕbulk = - (( dφ2dx2 + (2/self.grid.xs)*dφdx)*self.grid.dx)[self.grid.bulk_indcs]
		    
		    return Aϕbulk

		#Now run GMRES
		gmreskwargs = {'maxiter':int(1e3),'restart':int(100),'tol':1e-5}

		A_op_bulk = LinearOperator((self.grid.Nx-2, self.grid.Nx-2), matvec=Abulk_func)
		print("Approximate condition number: kappa={0:.3f}".format(self.estimate_κ_of_LinearOperator(A_op_bulk)))

		ϕe_bulk, code = gmres(A_op_bulk, get_b_bulk(),x0 = self.φe[self.grid.bulk_indcs], **gmreskwargs)
		#Set bulk phi from gmres
		self.ϕe[self.grid.bulk_indcs] = ϕe_bulk
		#Set bc for phi
		self.ϕe[0] = self.ϕe[1] #Zero E-field from electrons, plasma ions
		self.ϕe[-1] = self.ϕe[-2] + self.grid.dx* self.Z/self.R**2 #cancel E-field from central ion
		self.ϕe = self.ϕe - self.ϕe[-1]
		errs = np.abs(Abulk_func(self.ϕe[self.grid.bulk_indcs])-get_b_bulk())
		

		return np.mean(errs), code
	@staticmethod
	def estimate_κ_of_LinearOperator(A_operator):
		"""
		ChatGPT ftw!!
		"""

		# Find the largest and smallest singular values using the Arnoldi iteration
		largest_singular_value  = eigs(A_operator, k=1, which='LM', return_eigenvectors=False, tol=1e-2, maxiter = 1e4)
		print("Largest eigenvalue: " , np.abs(largest_singular_value))
		smallest_singular_value = eigs(A_operator, k=1, which='SM', return_eigenvectors=False, tol=1e-2, maxiter = 1e5)
		print("Smallest eigenvalue: ", np.abs(smallest_singular_value))

		# Calculate the approximate condition number
		condition_number = abs(largest_singular_value) / abs(smallest_singular_value)
		print("Number: ", condition_number, type(condition_number))
		return condition_number

	def update_ne(self, type='rel', alpha=1e-2):
		self.ne = self.small_update(self.get_ne_TF(self.ne, self.μ), self.ne, alpha=alpha)



	def small_update(self, f_new, f_old, type = 'both', alpha = 1e-3):
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

		self.steps = small_steps[0]
		return f_return

	def update_ne_W_Jacobi(self, alpha=1e-3, constant_hλ=False):
		"""
		Solves it using variable transformation of γ = r sqrt(n), or n = γ^2/r^2
		Solving γ'' + 1/(r^2 γ) (r γ' -γ)^2  (1+2 b/a) + 1/2 γ c/a = 0
		c =  
		"""
		#First define new variable γ
		γs = self.grid.xs * np.sqrt(self.ne) 
		
		#Create bc function for γ
		def γ_bc(γ):
			γ[0] = γ[1]/(1 + self.grid.dx*(-self.Z+ 1/self.grid.xs[1])) #Kato cusp condition
			γ[0] = (1/3*γ[3] - 3/2*γ[2] + 3*γ[1])/(11/6 + self.grid.dx*(-self.Z+ 1/self.grid.xs[1])) #Kato cusp condition
			# γ[-1]= 12/25*(4*γ[-2] -3*γ[-3] + 4/3*γ[-4]-1/4*γ[-5])/(1-self.grid.dx/self.R) # O(h^4) Approximate dn/dr=0
			# γ[-1]= γ[-2]/(1-self.grid.dx/self.R) 
			γ[-1]= -1/2*(-5* γ[-2] + 4*γ[-3] -1*γ[-4]) # γ''= 0 is nice...
			return γ
		γs = γ_bc(γs)
		#Update new variable, γ
		etas = self.TF.η_interp(self.ne)
		aW, bW = self.get_W_coeffs(etas, self.ne, self.T) #Get coefficients

		av_update = 0.5*(np.roll(γs,-1) + np.roll(γs,1))[self.grid.bulk_indcs]
		γprime    = self.grid.dfdx(γs)
		r = self.grid.xs
		
		#Interesting mid term from non-constant (finite-T) hλ
		if constant_hλ:
			w_part  = 0
		else:
			w_part  = 1/(r**2*γs) * (  r*γprime - γs)**2 * (1+2*bW/aW)

		if not self.ignore_vxc:
			c_part  = γs/(2*aW)*(self.T*etas - (self.μ + self.ϕe + self.ϕion - self.vxc_f(self.ne)) )
		if self.ignore_vxc:
			c_part  = γs/(2*aW)*(self.T*etas - (self.μ + self.ϕe + self.ϕion ))

		rho_update = 0.5*self.grid.dx**2 * (w_part + c_part)[self.grid.bulk_indcs]
		update = av_update + rho_update
		self.av_update = av_update
		self.rho_update = rho_update

		δγ = update - γs[1:-1]
		γs_new = np.zeros(self.grid.grid_shape)
		γs_new[self.grid.bulk_indcs] = γs[self.grid.bulk_indcs]*1 + alpha*δγ
		#Apply bc's on γ
		γs_new = γ_bc(γs_new)

		# Now remake ne 
		self.ne = γs_new**2/self.grid.xs**2

		# return γs_new**2/self.grid.xs**2
		print(np.linalg.norm(np.abs(1-self.ne/self.get_ne_TF(self.ne, self.μ)))/np.sqrt(self.grid.Nx))


	def update_ne_W_GMREs(self, alpha=1e-3, constant_hλ=False):
		"""
		Solves it using variable transformation of γ = r sqrt(n), or n = γ^2/r^2
		Solving γ'' + 1/(r^2 γ) (r γ' -γ)^2  (1+2 b/a) + 1/2 γ c/a = 0
		c =  
		"""

		# Define GMRES operators ###
		def γ_bc(γ):
			γ[0] = γ[1]/(1 + self.grid.dx*(-self.Z+ 1/self.grid.xs[1])) #Kato cusp condition
			γ[0] = (1/3*γ[3] - 3/2*γ[2] + 3*γ[1])/(11/6 + self.grid.dx*(-self.Z+ 1/self.grid.xs[1])) #Kato cusp condition
			γ[-1]= 12/25*(4*γ[-2] -3*γ[-3] + 4/3*γ[-4]-1/4*γ[-5])/(1-self.grid.dx/self.R) # O(h^4) Approximate dn/dr=0
			return γ

		def get_b_bulk():
			"""
			For Krylov GMRES optimization, Ax=b. This is b part, for Poisson Eqn.
			Args: None
			Returns: 
				1-D vectorlike b: Same shape
			"""
			b_bulk = np.zeros(self.grid.Nx-2)
			γbulk = (self.grid.xs * np.sqrt(self.ne))[self.grid.bulk_indcs]

			γs = np.zeros(self.grid.grid_shape)
			γs[self.grid.bulk_indcs] = γbulk
			#Set bc of γ
			γs = γ_bc(γs)
			
			r = self.grid.xs
			n = γs**2/r**2
			etas = self.TF.η_interp(n)
			aW, bW = self.get_W_coeffs(etas, n, self.T) #Get coefficients
			if not self.ignore_vxc:
				c_part  = γs/(2*aW)*(self.T*etas - (self.μ + self.ϕe + self.ϕion - self.vxc_f(self.ne)) )
			if self.ignore_vxc:
				c_part  = γs/(2*aW)*(self.T*etas - (self.μ + self.ϕe + self.ϕion ))

			b_bulk = -c_part[self.grid.bulk_indcs]

			return b_bulk.flatten()

		def Abulk_func( γbulk): 
			"""
			For Krylov GMRES optimization, Ax=b. This is A part, for Poisson Eqn.
			A(v) simulates A@v matrix multiplication 
			"""			
			γbulk = γbulk.copy()
			γbulk = np.where(np.abs(γbulk)<1e-8, 1e-8, γbulk)
			#Set bulk γ
			γs = np.zeros(self.grid.grid_shape)
			γs[self.grid.bulk_indcs] = γbulk
			#Set bc of γ
			γs = γ_bc(γs)

			#Make A matrix
			Aγbulk = self.grid.zeros.copy()[self.grid.bulk_indcs]


			r = self.grid.xs
			n = γs**2/r**2

			etas = self.TF.η_interp(n)
			
			aW, bW = self.get_W_coeffs(etas, n, self.T) #Get coefficients
			γprime    = self.grid.dfdx(γs)

			#Interesting mid term from non-constant (finite-T) hλ
			if constant_hλ:
				w_part  = 0
			else:
				w_part  = 1/(r**2*γs) * (  r*γprime - γs)**2 * (1+2*bW/aW)

			# if not self.ignore_vxc:
			# 	c_part  = γs/(2*aW)*(self.T*etas - (self.μ + self.ϕe + self.ϕion - self.vxc_f(self.ne)) )
			# if self.ignore_vxc:
			# 	c_part  = γs/(2*aW)*(self.T*etas - (self.μ + self.ϕe + self.ϕion ))

			dγdx   = self.grid.dfdx(γs)
			dγ2dx2 = self.grid.dfdx(dγdx)
			Aγbulk = (dγ2dx2 + w_part ) [self.grid.bulk_indcs]	
			return Aγbulk

		#Now run GMRES
		gmreskwargs = {'maxiter':int(2),'restart':int(50),'tol':1e-5}

		A_op_bulk = LinearOperator((self.grid.Nx-2, self.grid.Nx-2), matvec=Abulk_func)
		γ0 = (self.grid.xs * np.sqrt(self.ne))[self.grid.bulk_indcs]

		γ_bulk, code = gmres(A_op_bulk, get_b_bulk(), x0 = γ0, **gmreskwargs)

		#Set bulk γ from gmres
		γs_new = np.zeros(self.grid.grid_shape)
		γs_new[self.grid.bulk_indcs] = γ_bulk 
		
		#Apply bc's on γ
		γs_new = γ_bc(γs_new)
		
		# Now remake ne 
		self.ne = γs_new**2/self.grid.xs**2
		
		errs = np.linalg.norm(np.abs(1-self.ne/self.get_ne_TF(self.ne, self.μ)))/np.sqrt(self.grid.Nx)
		
		return np.mean(errs), code


	def update_ne_W_NewtonKrylov(self, alpha=1e-3, constant_hλ=False):
		
		def nfunc_to_min(n):

			etas = self.TF.η_interp(n)
			aW, bW = self.get_W_coeffs(etas, self.ne, self.T)
			grad_n = self.grid.dfdx(n)
			laplace_ne = 1/self.grid.xs**2 * self.grid.dfdx(self.grid.xs**2 * grad_n)
			W = aW/self.ne * laplace_ne + bW/self.ne**2 * grad_n**2

			eta = (self.μ + self.ϕe + self.ϕion - W - self.vxc_f(n))/self.T 
			

			return n - self.TF.n_free_TF(self.T, eta, self.μ)

		def γ_bc(γ):
			γ[0] = γ[1]/(1 + self.grid.dx*(-self.Z+ 1/self.grid.xs[1])) #Kato cusp condition
			γ[0] = (1/3*γ[3] - 3/2*γ[2] + 3*γ[1])/(11/6 + self.grid.dx*(-self.Z+ 1/self.grid.xs[1])) #Kato cusp condition
			# γ[-1]= 12/25*(4*γ[-2] -3*γ[-3] + 4/3*γ[-4]-1/4*γ[-5])/(1-self.grid.dx/self.R) # O(h^4) Approximate dn/dr=0
			γ[-1]= -1/2*(-5* γ[-2] + 4*γ[-3] -1*γ[-4]) # γ''= 0 is nice...
			return γ

		def γfunc_to_min(γbulk):		
		#Create bc function for γ
			γs = np.zeros(self.grid.Nx)
			γs[1:-1] = γbulk
			γs = γ_bc(γs)
			r = self.grid.xs

			ne = γs**2/r**2
			
			#Update new variable, γ
			etas = self.TF.η_interp(ne)
			aW, bW = self.get_W_coeffs(etas, ne, self.T) #Get coefficients

			γprime = self.grid.dfdx(γs)
			γprimeprime =self.grid.dfdx(γprime)
			
			#Interesting mid term from non-constant (finite-T) hλ
			if constant_hλ:
				w_part  = 0
			else:
				w_part  = 1/(r**2*γs) * (  r*γprime - γs)**2 * (1+2*bW/aW)

			if not self.ignore_vxc:
				c_part  = γs/(2*aW)*(self.T*etas - (self.μ + self.ϕe + self.ϕion - self.vxc_f(self.ne)) )
			if self.ignore_vxc:
				c_part  = γs/(2*aW)*(self.T*etas - (self.μ + self.ϕe + self.ϕion ))

			f = (γprimeprime + w_part + c_part)[1:-1]
			return f		
			
		
		# from scipy.optimize import root

		# from scipy.optimize import InverseJacobian
		# def preconditioner(n):


		# self.n = root(func_to_min, self.ne.copy(), method='broyden1', options={'maxiter':1})
		# γbulk = newton_krylov(γfunc_to_min, (self.grid.xs*np.sqrt(self.ne))[1:-1], inner_maxiter=1000, iter=5, outer_k=50)
		γbulk = fsolve(γfunc_to_min,  (self.grid.xs*np.sqrt(self.ne))[1:-1])
		γs = np.zeros(self.grid.Nx)
		γs[1:-1] = γbulk
		γs = γ_bc(γs)
		
		self.ne = γs**2/self.grid.xs**2
		print("Err: ", np.linalg.norm(np.abs(1-self.ne/self.get_ne_TF(self.ne, self.μ)))/np.sqrt(self.grid.Nx))
		
	
	def update_bound_free(self, φ_shift=0):
		"""
		Gets bound free separation using approximation  in ThomasFermi. 
		Only iteratively updates Zstar
		Returns: 
			Exact Zstar using bound, free.

		"""

		etas = self.get_eta_from_sum(self.ne, self.μ)

		self.n_f = ThomasFermi.n_free_TF (self.T, etas, self.μ + φ_shift)
		self.n_b = ThomasFermi.n_bound_TF(self.T, etas, self.μ + φ_shift)


		new_Zstar = self.Z - self.grid.integrate_f(self.n_b)


		self.Zstar = self.Zstar + 1e-2*(new_Zstar-self.Zstar) #smaller update
		return new_Zstar

	def compensate_ρe(self):#, delta_Q):
		"""
		After Zstar updates, Q changes by alot. To get better initial guess for this system,
		We increase the electron density to remain overall neutrality and get better numerics.
		"""
		delta_ne = self.get_Q()/self.Vol #delta_Q/self.Vol 
		self.ne += delta_ne  #Increase by delta_Q overall 

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
	

	def solve_TF(self, verbose=False, nmax = 1e3, tol=1e-4):
		"""
		Solve TF OFDFT equation, assuming a given Zbar for the plasma ions 
		"""
		print("\nBeginning TF NPA loop")
		print("_________________________________")
		print("Initial Q = ", self.get_Q()) 

		Q_list, poisson_err_list, change_list = [], [], []
		Q=0
		converged = False
		n = 0
		while not converged and n < int(nmax):
		# while  n < int(nmax):
			Q_old = Q
			old = self.μ, np.mean(self.ne), np.mean(self.ϕe) 
			
			# Update physics in this order
			poisson_err, gmres_code = self.update_φe()

			if self.TFW:
				# self.update_ne_W_Jacobi(alpha=0.5, constant_hλ=False)
				self.update_ne_W_NewtonKrylov()
				# self.update_ne(alpha=1e-3)
				#Update Chemical potential
			else:
				self.update_ne(alpha=1e-2)
				self.compensate_ρe()
			if np.abs(Q)<5e-3 or np.abs(Q)>0.5:
				self.set_μ_TF()
			else: 
				self.update_μ_newton(alpha1=1)
				# self.update_μ_newton(alpha1=.1)
				# # self.shift_μ()
			

			rho_err = np.linalg.norm(np.abs(1-self.ne/self.get_ne_TF(self.ne, self.μ)))/np.sqrt(self.grid.Nx)
			# Save convergence variables
			Q = self.get_Q()
			delta_Q = self.get_Q() - Q_old

			new = self.μ, np.mean(self.ne), np.mean(self.ϕe) 
			change  = self.L2_change(new,old)
			Q_list.append(Q); poisson_err_list.append(poisson_err); change_list.append(change)

			if verbose and (n%10==0 or n<10):
				print("__________________________________________")
				print("TF Iteration {0}".format(n))
				print("	Gmres code: {0}".format(gmres_code))
				print("	mu = {0:10.4f}".format(self.μ))	
				print("	Q  = {0:10.3e}, Poisson Err = {1:10.3e}, rho Err = {2:10.3e}".format(Q, poisson_err, rho_err))
				print("	Change = {0:10.3e}".format(change))
				# self.make_plots(show=False)
			
			# Converged??
			if abs(Q)<5e-4 and abs(poisson_err<1e-4) and abs(rho_err)<1e-6:
				converged=True
			n+=1
			#self.make_plots(True)
		# print("Done, creating convergence and AA plots.")
		# self.make_plots(show=False)
		# self.convergence_plot([np.abs(Q_list), poisson_err_list, change_list], [r"|Q|","Mean Poisson Error", "L2 change (mu, rho, phi)"])
		return converged

	def solve_NPA(self,verbose=True):
		print("Starting NPA Loop")
		converged = False
		n, nmax = 0, int(1e2	)
		while not converged and n < nmax:
			TF_code = self.solve_TF(verbose=True) #Using Zstar, computes
			
			Zstar_old, Q_old = self.Zstar, self.get_Q()

			φ_shift = -self.vxc_f(self.ne)[-1] # Can escape all potentials up to R
			# φ_shift =  self.ϕe[self.rws_index] + self.ϕion[self.rws_index] -self.vxc_f(self.ne)[self.rws_index] # Can escape rs

			self.update_bound_free(φ_shift = φ_shift ) #Updates n_b, n_f, Zstar
			self.make_ρi() #Updates ion density, automatically plugged into Poisson Eqn.
			# delta_Q = self.get_Q() - Q_old
			self.compensate_ρe() #Adds constant electron density so as to compensate change in Q
			self.set_μ_TF()


			if verbose==True:
				print("__________________________________________")
				print("__________________________________________")
				print("NPA Iteration {0}".format(n))
				print("Z* = {0:3e}".format(self.Zstar))
				Zstar_err = abs(1-Zstar_old/self.Zstar)
				print("Z* rel change = {0:4e}".format(Zstar_err))

			if np.abs(Zstar_err) < 1e-6:
				converged = True

			n+=1


	###############	
	## PLOTTING ###
	###############
	def convergence_plot(self, vars, var_names):
		fig, ax = plt.subplots(ncols=1, figsize=(10,8),facecolor='w')
		for var, name in zip(vars,var_names):
			ax.plot(var, '--.', label=name)

		ax.set_xlabel("Iteration",fontsize=20)
		ax.set_ylabel("Error",fontsize=20)
		ax.tick_params(labelsize=20,which='both')
		#ax.set_yscale('symlog',linthresh=1e-3)
		ax.set_yscale('log')
		ax.set_ylim(1e-5,10)
		ax.grid(which='both')
		ax.legend(fontsize=15)

		plt.tight_layout()

		#Save
		name = "NPA_convergence.png"
		plt.savefig("/home/zach/plasma/atomic_forces/average_atom/media/" + name, dpi=300, bbox_inches='tight',facecolor="w")
		#plt.show()


	def make_plots(self, show=False):

		fig, axs = plt.subplots(ncols=2,figsize=(20,8),facecolor='w')
  	      
  	      # Potential φ plot
		axs[0].plot(self.grid.xs , self.φion, label=r"$\phi_{ion}$")
		axs[0].plot(self.grid.xs , -self.φe, label=r"$-\phi_{e}$")
		axs[0].plot(self.grid.xs , self.φe + self.φion, label=r"$\phi$")
		if not self.ignore_vxc:
			axs[0].plot(self.grid.xs , -self.vxc_f(self.ne) , label=r"$-\delta V_{xc}/\delta n_e$")
			axs[0].plot(self.grid.xs , self.φe + self.φion - self.vxc_f(self.ne) , label=r"$\phi -	 \delta V_{xc}/\delta n_e$")


		axs[0].set_ylabel(r'$\phi$ [A.U.]',fontsize=20)
		axs[0].set_ylim(-1e-1,1e6)
		axs[0].set_yscale('symlog',linthresh=1e-3)

		# Density ne plot
		axs[1].plot(self.petrov.r_data, self.petrov.rho_data + self.petrov.rho_0, 'k--', label="Petrov AA")
		axs[1].plot(self.grid.xs, self.ne , label=r'$n_e$')
		axs[1].plot(self.grid.xs, self.get_ne_TF(self.ne, self.μ) , label=r'$\frac{\sqrt{2}}{\pi^2}T^{3/2}\mathcal{I}_{1/2}(\eta)$')
		axs[1].plot(self.grid.xs, self.n_b, label=r'$n_b$')
		axs[1].plot(self.grid.xs, self.n_f, label=r'$n_f$')
		axs[1].plot(self.grid.xs, self.ρi, label=r'$ Z^\ast n^0_i g_{ii}(r) $ ')
		axs[1].plot(self.grid.xs, np.abs(self.ρi - self.ne), label=r'$|\Sigma_j \rho_j|$ ')
		
		axs[1].set_ylabel(r'$n$ [A.U.]',fontsize=20)
		axs[1].set_ylim(1e-3,1e6)
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
			text = ("{0}, {1}\n".format(self.name, self.aa_type.replace("_", " "))+ 
				r"$r_s$ = " + "{0:.2f},    ".format(self.rs,2) +
				r"$R_{NPA}$ = " + "{0}\n".format(self.R)  +
        			r"$T$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.T, self.T/eV) +
        			r"$\mu$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.μ, self.μ/eV) +
        			r"$\mu-V(R)$ = " + "{0:.3f} [A.U.] = {1:.2f} eV\n".format(self.μ-self.vxc_f(self.ne)[-1],(self.μ-self.vxc_f(self.ne)[-1])/eV) +
        			r"$Z^\ast =$ " + "{0:.2f}".format(self.Zstar)  )
			props = dict(boxstyle='round', facecolor='w')
			ax.text(0.05,0.95, text, fontsize=15, transform=ax.transAxes, verticalalignment='top', bbox=props)

		plt.tight_layout()
		name = "NPA_{0}_rs{1}_{2}eV_R{3}.png".format(self.name, np.round(self.rs,2), np.round(self.T/eV,2) ,np.round(self.R))
		plt.savefig("/home/zach/plasma/atomic_forces/average_atom/media/" + name, dpi=300, bbox_inches='tight',facecolor="w")
		if show == True:
			plt.show()

	def make_plot_bound_free(self, show=False):

		fig, axs = plt.subplots(ncols=2,figsize=(20,8),facecolor='w')
  	      
		# Density * 4pi r^2 plot
		factor = 4*np.pi*self.grid.xs**2
		axs[0].plot(self.petrov.r_data, 4*np.pi*self.petrov.r_data**2*(self.petrov.rho_data + self.petrov.rho_0), 'k--', label="Petrov AA")
		axs[0].plot(self.grid.xs, self.ne*factor ,'k',label=r'$n_e$')
		axs[0].plot(self.grid.xs, self.n_b*factor, label=r'$n_b$')
		axs[0].plot(self.grid.xs, self.n_f*factor, label=r'$n_f$')
		
		
		axs[0].set_ylabel(r'$4 \pi r^2 n_e(r) $ [A.U.]',fontsize=20)
		axs[0].set_ylim(1e-2, 1e3)
		axs[0].set_yscale('log')

		# Density ne plot
		axs[1].plot(self.petrov.r_data, self.petrov.rho_data + self.petrov.rho_0, 'k--', label="Petrov AA")
		axs[1].plot(self.grid.xs, self.ne , 'k', label=r'$n_e$')
		axs[1].plot(self.grid.xs, self.n_b, label=r'$n_b$')
		axs[1].plot(self.grid.xs, self.n_f, label=r'$n_f$')
		axs[1].plot(self.grid.xs, self.ρi, label=r'$ Z^\ast n^0_i g_ii(r) $ ')
		axs[1].plot(self.grid.xs, np.abs(self.ρi - self.ne), label=r'$|\Sigma_j \rho_j|$ ')

		axs[1].set_ylabel(r'$n_e$ [A.U.]',fontsize=20)
		axs[1].set_ylim(1e-3,1e6)
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
        			r"$T$ = " + "{0} [A.U.] = {1} eV\n".format(np.round(self.T,2),np.round(self.T/eV,2)) + r"$\mu$ = " + "{0} [A.U.]\n".format(np.round(self.μ,2)) +
        			r"$Z^\ast = $" + "{0}".format(np.round(self.Zstar,2))  )

			props = dict(boxstyle='round', facecolor='w')
			ax.text(0.05,0.95, text, fontsize=15, transform=ax.transAxes, verticalalignment='top', bbox=props)

		plt.tight_layout()
		name = "NPA_densities_{0}_rs{1}_{2}eV_R{3}.png".format(self.name, np.round(self.rs,2), np.round(self.T/eV,2) ,np.round(self.R))
		plt.savefig("/home/zach/plasma/atomic_forces/average_atom/media/" + name, dpi=300, bbox_inches='tight',facecolor="w")
		if show == True:
			plt.show()



if __name__=='__main__':
	# 				 Z    A   T[AU] rs  R
	# atom = NeutralPseudoAtom(13, 28, 1050*Kelvin,  3.1268 , 10, Npoints=1000,name='Aluminum', TFW=False, ignore_vxc=True)
	atom = NeutralPseudoAtom(13, 28, 1*eV,  3 , 30, Npoints=3000, Zstar_init=3.8, name='Aluminum', TFW=False, ignore_vxc=False)

	# folder= "/home/zach/plasma/atomic_forces/average_atom/data/"
	# fname = "Aluminum_NPA_TFD_R9.0e+00_rs3.0e+00_T3.7e-02eV_Zstar3.8.dat"
	# atom  = load_NPA(folder + fname,TFW=False, ignore_vxc=False)
	#atom.solve_TF(verbose=True)
	atom.solve_NPA(verbose=True)
	
	# atom.solve_TF(verbose=True)
	atom.save_data()
	atom.make_plots()
	
	# print(atom.TF.n_TF(1,1e-20))