import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq, dst, idst


from atomic_forces.fdints import fdints
from atomic_forces.GordonKim.python.atoms import Potentials


Iminusonehalf  = lambda eta: gamma(3/2)*fdints.fminusonehalf(float(eta))
Ionehalf   = lambda eta: gamma(3/2)*fdints.fonehalf(float(eta))
Ithreehalf = lambda eta: gamma(3/2)*fdints.fthreehalf(float(eta))

π = np.pi
eV=0.0367512

import matplotlib.pyplot as plt

class response():


	def __init__(self, atom, dst_type=3, N=1000, ρtype='data',χtype='Lindhard', linear_type='standard'):
		self.atom = atom
		self.T = atom.T # No * or / eV right???
		self.rs = atom.rs
		self.R = atom.R

		#self.r_list = np.concatenate( [np.geomspace(1e-5,1,num=100) , np.linspace(1.01,100,num=500)])
		self.N = N
	
		self.dst_type=dst_type
		if  dst_type==1: 
			self.r_list = np.linspace(0, self.R, num=self.N+1)[1:]
			self.dr = self.r_list[1]-self.r_list[0]
			self.k_list = np.array([π*(l+1)/self.R for l in range(self.N)] ) #Type 1
		elif dst_type==2:
			self.r_list = np.linspace(0, self.R, num=self.N+1)[1:]
			self.dr = self.r_list[1]-self.r_list[0]
			self.r_list -= self.dr/2 #So now theoretically r_list[1/2] would be zero
			self.k_list = np.array([π*(l+1)/self.R for l in range(self.N)] ) #Type 1
		elif dst_type==3:	
			self.r_list = np.linspace(0, self.R, num=self.N+1)[1:]
			self.dr = self.r_list[1]-self.r_list[0]
			self.k_list = np.array([π*(l+0.5)/self.R for l in range(self.N)] ) #Type 3
		elif dst_type==4:		
			self.r_list = np.linspace(0, self.R, num=self.N+1)[1:]
			self.dr = self.r_list[1]-self.r_list[0]
			self.r_list -= self.dr/2 #So now theoretically r_list[1/2] would be zero
			self.k_list = np.array([π*(l+0.5)/self.R for l in range(self.N)] ) #Type 3

		self.vol_rs = 4/3*π*self.rs**3
		self.vol_R = 4/3*π*self.R**3
		self.rho_av = self.atom.ne_bar


		self.make_constants()
		self.make_ρ()
		self.make_χ(type=χtype)
		self.make_potentials(type=linear_type)
		#self.add_neutrality()

	##### DENSITIES #####
	# def δρf_data(self, r):
		
		
	# 	return δρ


	##### LINEAR RESPONSES ###
	def χ_TF(self,k):
		χ_lowk = +k**2/(4*π)* self.kTF**2/(self.kTF**2 + k**2)  # Low-k full but full renormalized χ
		vee = 4*π/k**2
		χ_lowk =  -1/( vee + 4*π/self.kTF**2)  # rewritten
		return χ_lowk

	def χ_GP(self,k): # George Petrov
		λ=0.11
		χ0 = self.kF/π**2 * 1/ (1 + 3*λ*(0.5*k/self.kF)**2 )

		χ_renorm = -χ0/(1 + χ0* 4*π/k**2) 
		return χ_renorm

	def χ_TFW(self,k): #Stanton-Murillo
		vee = 4*π/k**2
		λ = 1/9 #TF GC limit, λ=1 is traditional Weizsacker

		η0 = Potentials.find_eta(self.rho_av, self.T)
		ν = np.sqrt(8/self.T)/(3*π) * λ*Iminusonehalf(η0)

		χ_lowk = -1/( vee +4*π/self.kTF**2 + π*ν*k**2/self.kTF**4)  # rewritten
		return χ_lowk

	def γ0_param(self):			
		N = lambda Θ: 1+ 2.8343*Θ**2 - 0.2151*Θ**3 + 5.2759*Θ**4
		D = lambda Θ: 1+ 3.9431*Θ**2 + 7.9138*Θ**4


		h = lambda Θ: N(Θ)/D(Θ) * np.arctanh(1/Θ)
		hprime = lambda Θ: (h(Θ + 1e-6)-h(Θ))/1e-6
		γ0 = 1/8/self.T * self.Θ * ( h(self.Θ) - 2*self.Θ*hprime(self.Θ) )
		print(self.Θ, h(self.Θ), hprime(self.Θ))
		return γ0

	def χ_TFWLFC(self,k): #Stanton-Murillo
		vee = 4*π/k**2
		λ = 1/9 #TF GC limit, λ=1 is traditional Weizsacker

		η0 = Potentials.find_eta(self.rho_av, self.T)
		ν = np.sqrt(8/self.T)/(3*π) * λ*Iminusonehalf(η0)
		γ0 = self.γ0_param()
		#print("γ0=",γ0)

		χ_lowk = -1/( vee +4*π*(1/self.kTF**2 - γ0 ) + π*ν*k**2/self.kTF**4)  # rewritten
		return χ_lowk

	def χ_Lindhard(self,k):
	    ktilde = k/(2*self.kF) 
	    Ndos = self.kF/π**2
	    χ0 = Ndos *(1/2 + 1/(4*ktilde)*(1-ktilde**2)*np.log(abs( (ktilde+1)/(ktilde-1) )) )#Lindhard
	    
	    χ_renorm = -χ0/(1 + χ0* 4*π/k**2) 
	    return χ_renorm
	
	#### MAKE/SET ATTRIBUTES ###
	def make_constants(self):
		self.n0 = self.atom.ne_bar
		self.EF = 0.5*(3*π**2*self.n0)**(2/3)
		self.kF = (2*self.EF)**(1/2)
		self.kTF = np.sqrt(  4*π* self.atom.Zstar*self.n0  /np.sqrt(self.T**2 + (2/3*self.EF)**2)  )
		self.Θ = self.T/self.EF

	def make_ρ(self):
		"""
		Create the density fluctuation used in linear response from either the analytic fit or the data 
		"""
		# Number density
		# 	Make r-space
		δnf = self.atom.δn_f
		δnf_func   =  interp1d(self.atom.grid.xs, δnf, bounds_error = False, fill_value = (δnf[0], δnf[-1]) )#'extrapolate')
		# δnf_func  =  lambda r: nf_func(r) - nf[-1]# - n0 # Free change from uniformity
		self.δnf  =  δnf_func(self.r_list)

		# 	Make k-space
		integrand = 4*np.pi* self.r_list* self.δnf
		δn_dst = 0.5*self.dr*dst(integrand,type=self.dst_type)/self.k_list
		self.δnf_tilde = np.array(δn_dst)

		# Charge density
		self.δρf       = - self.δnf  
		self.δρf_tilde = - self.δnf_tilde
		


	def make_χ(self,type='Lindhard'):	
		if type=='Lindhard':
			self.χs = np.array([self.χ_Lindhard(k) for k in self.k_list])
		elif type=='TF':
			self.χs = np.array([self.χ_TF(k) for k in self.k_list])
		elif type=='TFW':
			self.χs = np.array([self.χ_TFW(k) for k in self.k_list])
		elif type=='GP':
			self.χs = np.array([self.χ_GP(k) for k in self.k_list])


	def make_potentials(self, type="standard"):
		"""
		Currently working with vii(k) = vii(k)^0 + δvii(k)
		for vii(k)^0 = -Zbar vei(k)
		δvii(k) = -Zbar δne v(k) for v(k)=4 π/k^2
		vii(k) = vii(k)^0 - Zbar δne v(k)
		"""
		if type=='standard':
			"""
			vii = 4 π Zstar^2 /k^2 + δρf^2/χ
			    = 4 π Zstar^2 /k^2 + χ vei^2
			"""
			self.vei_tilde  = - self.δρf_tilde/self.χs
			self.vii0_tilde =   4*π*self.atom.Zstar**2/self.k_list**2
			self.δvii_tilde =  -self.vei_tilde*(self.δρf_tilde)
		elif type=='Ichimaru':
			self.vei_tilde  = - self.δρf_tilde/self.χs
			self.vii0_tilde = - self.atom.Zstar * self.vei_tilde
			self.δvii_tilde =   self.atom.Zstar * self.δρf_tilde * 4*np.pi/self.k_list**2

		self.vii_tilde  =   self.vii0_tilde + self.δvii_tilde
		 
		self.vei = 2*self.N *1/(2*π)**3*0.5*(π/self.R)*idst(4*π*self.k_list*self.vei_tilde ,type=self.dst_type)/self.r_list
		self.vii = 2*self.N *1/(2*π)**3*0.5*(π/self.R)*idst(4*π*self.k_list*self.vii_tilde ,type=self.dst_type)/self.r_list

		self.vii0 = 2*self.N *1/(2*π)**3*0.5*(π/self.R)*idst(4*π*self.k_list*self.vii0_tilde ,type=self.dst_type)/self.r_list
		self.δvii = 2*self.N *1/(2*π)**3*0.5*(π/self.R)*idst(4*π*self.k_list*self.δvii_tilde ,type=self.dst_type)/self.r_list
	
	# def add_neutrality(self):
	# 	self.δρf_tilde = np.concatenate([[self.atom.Zstar], self.δρf_tilde])	
	# 	self.k_list = np.concatenate([[0.0],self.k_list])
	# 	self.χs = np.concatenate([[0], self.χs])		
	# self.vii_tilde = np.concatenate([[0], self.vii_tilde])
		# self.vei_tilde = np.concatenate([[0], self.vei_tilde])


	#########################################
	############# PLOTS #####################
	#########################################
	def check_ρ(self):
		#δρf_check = 1/(2*π)**3*0.5*(π/self.R)*dst(4*π*self.k_list*self.δρf_tilde ,type=1)/self.r_list
		#new_r_list = np.concatenate([self.r_list,[self.r_list[-1]+self.dr]])
		δρf_check = 2*self.N *1/(2*π)**3*0.5*(π/self.R)*idst(4*π*self.k_list*self.δρf_tilde ,type=self.dst_type)/self.r_list

		#err = np.mean((δρf_check-self.δρf))
		#err_ratio = np.mean((δρf_check-self.δρf)/np.sqrt(self.δρf**2 + δρf_check**2 ))
		#print("Mean Err: {0}, Err ratio: {1} ".format(err, err_ratio))
		print(self.δρf[-1])
		fig = plt.figure(2, figsize=(16,6),facecolor='w')

		gs = fig.add_gridspec(1,2)
		ax = fig.add_subplot(gs[0,0])

		ax.plot(self.r_list, δρf_check,label='After DST')
		ax.plot(self.r_list, self.δρf,'--',label='Original')
		ax.plot(self.r_list, self.r_list*self.δρf,'--',label=r'$r \rho(r)$')

		ax.set_ylabel(r'$\delta \rho$ [A.U]',fontsize=20)
		ax.set_xlabel(r'$r$ [A.U]',fontsize=20)
		ax.legend(fontsize=20)
		ax.tick_params(labelsize=20)
		#ax.set_yscale('log')
		#ax.set_ylim(-0.05,0.2)
		ax.set_yscale('symlog',linthresh=1e-2)
		ax.set_xscale('log')

		ax1 = fig.add_subplot(gs[0,1])
		ax1.plot(self.k_list, self.δρf_tilde,'--.', label=r'$\delta\tilde{\rho}$')
		ax1.plot(self.k_list, -self.atom.Zstar*np.ones(len(self.k_list)),'k-', label=r'$Z^*$')
		ax1.legend(fontsize=20)
		ax1.set_ylabel(r'$\delta\tilde{\rho}$ [A.U]',fontsize=20)
		ax1.set_xlabel(r'$k$ [A.U]',fontsize=20)
		ax1.tick_params(labelsize=20)
		#ax.set_yscale('log')
		#ax.set_ylim(-0.05,0.2)
		ax1.set_yscale('symlog',linthresh=1e-2)
		ax1.set_xscale('log')

		plt.grid(which='both',alpha=0.3)
		plt.tight_layout()
		plt.savefig("../media/ρ_of_r.png",dpi=300,bbox_inches='tight',facecolor="w")


	def compare_χs(self):
		fig, ax = plt.subplots(figsize=(8,6),facecolor='w')

		ks = np.geomspace(1e-3,1e3,num=100) 
		ax.plot(ks,[self.χ_Lindhard(k) for k in ks],'-',label='Lindhard')
		ax.plot(ks,[self.χ_TF(k) for k in ks],'-',label='TF')
		ax.plot(ks,[self.χ_TFW(k) for k in ks],'-',label='TFW')
		#ax.plot(ks,[self.χ_TFWLFC(k) for k in ks],'-',label='TFW LFC')
		ax.plot(ks,[self.χ_GP(k) for k in ks],'-',label='GP')
		ax.plot(ks,-ks**2/(4*π),'k--',label=r'$-k^2/4\pi$')
		ax.axvline(x = 2*self.kF, color = 'k',linestyle='-', label = r'$2 k_F$')


		ax.set_ylabel(r'$\chi(k)$ ',fontsize=20)
		ax.set_xlabel(r'$k$ [A.U.]',fontsize=20)
		ax.legend(fontsize=20)
		ax.tick_params(labelsize=20)
		ax.set_xscale('log')
		ax.set_yscale('symlog',linthresh=1e-5)
		ax.set_ylim(-1,-1e-4)
		#ax.set_xlim(0.5,1e3)
		plt.grid(which='both',alpha=0.3)
		plt.tight_layout()
		plt.savefig("../media/chi_compare.png",dpi=300,bbox_inches='tight',facecolor="w")

