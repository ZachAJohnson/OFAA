import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import gamma

from ..fdints import fdints
import pylibxc

import matplotlib.pyplot as plt

from hnc.hnc.constants import *

eV=0.0367512
π = np.pi



def Fermi_Energy(ne):
    E_F = 1/(2*m_e) * (3*π**2 * ne)**(2/3)
    return E_F

def Fermi_velocity(ne):
    v_F = np.sqrt(2*Fermi_Energy(ne)/m_e)
    return v_F

def Fermi_wavenumber(ne):
    k_F = Fermi_velocity(ne)*m_e
    return k_F

def Degeneracy_Parameter(Te, ne):
    θ = Te/Fermi_Energy(ne)
    return θ

def Gamma(T, n, Z):
    β = 1/T
    rs = rs_from_n(n)
    return Z**2*β/rs

def Debye_length(T, ni, Zbar):
    ne = Zbar*ni
    EF = Fermi_Energy(ne)
    T_effective = (T**1.8 + (2/3*EF)**1.8)**(1/1.8)
    λD = 1/np.sqrt(  4*π*ne/T_effective ) # Including degeneracy
    # λD = 1/np.sqrt(  4*π*ne/T_effective + 4*π*Zbar**2*ni/T  )  # Including ions
    return λD

def Kappa(T, ni, Zbar):
    rs = rs_from_n(ni)
    λD = Debye_length(T, ni, Zbar)
    return rs/λD

def n_from_rs( rs):
    """
    Sphere radius to density, in any units
    """
    return 1/(4/3*π*rs**3)

def rs_from_n(n):
    """
    Density to sphere radius, in any units.
    """
    return (4/3*π*n)**(-1/3)

def find_η(Te, ne):
    """
    Gets chemical potential in [AU] that gives density ne at temperature Te
    """
    f_to_min = lambda η: ThomasFermi.n_TF(Te, η)-ne # η = βμ in ideal gas, no potential case
    
    root_and_info = root(f_to_min, 0.4,tol=1e-4)

    η = root_and_info['x'][0]
    return η

def P_Ideal_Fermi_Gas(Te, ne):
    """
    Gets the noninteracting pressure in AU.
    """
    η = find_η(Te, ne)
    Ithreehalf = FermiDirac.Ithreehalf(η)
    Θ = Degeneracy_Parameter(Te, ne)
    P = Te * ne * Θ**(3/2) * Ithreehalf # Goes to 2/5 EF ne
    return P

def E_Ideal_Fermi_Gas(Te, ne):
    """
    Gets the noninteracting pressure in AU.
    """
    η = find_η(Te, ne)
    Ithreehalf = FermiDirac.Ithreehalf(η)
    Θ = Degeneracy_Parameter(Te, ne)
    E = 3/2 * Te * Θ**(3/2) * Ithreehalf # Goes to 3/5 EF
    return E

def More_TF_Zbar( Z, n_AU, T_AU):
        """
        Finite Temperature Thomas Fermi Charge State using 
        R.M. More, "Pressure Ionization, Resonances, and the
        Continuity of Bound and Free States", Adv. in atomic 
        Mol. Phys., Vol. 21, p. 332 (Table IV).

        Z = atomic number
        num_density = number density (1/cc)
        T = temperature (eV)
        """

        alpha = 14.3139
        beta = 0.6624
        a1 = 0.003323
        a2 = 0.9718
        a3 = 9.26148e-5
        a4 = 3.10165
        b0 = -1.7630
        b1 = 1.43175
        b2 = 0.31546
        c1 = -0.366667
        c2 = 0.983333

        n_cc = n_AU*AU_to_invcc
        T_eV = T_AU*AU_to_eV

        convert = n_cc*1.6726e-24
        R = convert/Z
        T0 = T_eV/Z**(4./3.)
        Tf = T0/(1 + T0)
        A = a1*T0**a2 + a3*T0**a4
        B = -np.exp(b0 + b1*Tf + b2*Tf**7)
        C = c1*Tf + c2
        Q1 = A*R**B
        Q = (R**C + Q1**C)**(1/C)
        x = alpha*Q**beta

        return Z*x/(1 + x + np.sqrt(1 + 2.*x))
        

### LINEAR RESPONSE FUNCTIONS #####
def χ0_Lindhard(k, kF):
    ktilde = k/(2*kF) 
    Ndos = kF/π**2
    χ0 = Ndos *(1/2 + 1/(4*ktilde)*(1-ktilde**2)*np.log(abs( (ktilde+1)/(ktilde-1) )) )#Lindhard
    return -χ0

def G_SLFC(k, kF):
    """
    Taylor 1978 - https://iopscience.iop.org/article/10.1088/0305-4608/8/8/011/pdf
    Rewritting their math I get
    χ = χ0/(1-χ0 Uee) for Uee = vee (1-G) and vee = 4π/k**2
      = -Π0/(1 + vee(1-k**2 φ)Π0)
      
    Unforunately, I'm not sure of the sign between χ0 and Π0, so I'm not sure of the overall sign, but G is fine.
    """
    λ = (π*kF)**-1
    φ0 = (1 + 0.1534 * λ)/(4*kF**2)
    G = k**2*φ0
    return G
    
def χ_Lindhard(k, kF):
    χ0 = χ0_Lindhard(k, kF)
    vee = 4*π/k**2
    G = G_SLFC(k, kF)
    Uee = vee*(1-G)
    χ_renorm = χ0/(1 - χ0* Uee) 
    return χ_renorm

def χ_TF(k, kTF):
    χ_lowk = +k**2/(4*π)* kTF**2/(kTF**2 + k**2)  # Low-k full but full renormalized χ
    vee = 4*π/k**2
    χ_lowk =  -1/( vee + 4*π/kTF**2)  # rewritten
    return χ_lowk



class FermiDirac():

	def __init__(self):
		"""

		"""

	@staticmethod
	@np.vectorize
	def Fminusonehalf(eta):
		return fdints.fminusonehalf(float(eta))

	@staticmethod
	@np.vectorize
	def Fonehalf(eta):
		return fdints.fonehalf(float(eta))

	@staticmethod
	@np.vectorize
	def Fthreehalf(eta):
		return fdints.fthreehalf(float(eta))

	@staticmethod
	@np.vectorize
	def Fminusonehalfprime(eta):
		eps = eta*1e-6
		return (FermiDirac.Fminusonehalf(eta + eps) - FermiDirac.Fminusonehalf(eta-eps))/(2*eps)

	@staticmethod
	@np.vectorize
	def Iminusonehalf(eta):
		l=-1/2
		return gamma(l+1)*FermiDirac.Fminusonehalf(eta)

	@staticmethod
	@np.vectorize
	def Ionehalf(eta):
		# l=1/2
		# return gamma(l+1)*FermiDirac.Fonehalf(eta)
		# return np.sqrt(np.pi)/2*FermiDirac.Fonehalf(eta)
		return 0.886227*FermiDirac.Fonehalf(eta)
		

	@staticmethod
	@np.vectorize
	def Ithreehalf(eta):
		l=3/2
		return gamma(l+1)*FermiDirac.Fthreehalf(eta)

	@staticmethod
	@np.vectorize
	def Iminusonehalfprime(eta):
		l=-1/2
		return gamma(l+1)*FermiDirac.Fminusonehalfprime(eta)
	
	@staticmethod
	@np.vectorize
	def inc_lower_Ionehalf(xmid, eta):
		"""
		Accurate to <1e-4 for eta>1e4, better (<1e-6 for below.)
		Bad relative error for eta<1e-2, but great absolute error 
		"""
		if 1e2 + eta<xmid: # So xmid effectively infinite compared to eta
			return FermiDirac.Ionehalf(eta)
		else:
			integrand = lambda x:  (x**0.5/(np.exp(x-eta) + 1)).real
			integral, info = quad(integrand, 0, xmid)
			return integral

	@staticmethod
	@np.vectorize
	def inc_upper_Ionehalf(xmid, eta): 
		"""
		Accurate to <1e-4 for eta>1e4, better (<1e-6 for below.)
		Bad relative error for eta<1e-2, but great absolute error 
		"""
		integrand = lambda x:  (x**0.5/(np.exp(x-eta) + 1)).real
		x_infinite = 2*eta+1e2 # max(eta*100,1e3) #Upper limit (approx infinity)
		integral, info = quad(integrand, xmid, x_infinite, limit=50, epsabs=1e-15, epsrel=1e-5) # x0 to infinity
		return integral




class ThomasFermi():
	def __init__(self, T, ignore_vxc = False):
		self.T = T
		self.η_interp = self.make_η()
		self.η_interp = np.vectorize(self.η_interp)

		if ignore_vxc == False:
			self.vxc_func = self.fast_vxc()
		self.hλ_func = self.fast_hλ()
		
		self.hλprime_func = lambda η: (self.hλ_func(η*(1+1e-5))-self.hλ_func(η*(1-1e-5)))/(2*1e-5)
		self.hλprime_func = np.vectorize(self.hλprime_func)


	@staticmethod	
	@np.vectorize
	def n_TF(T, eta):
		return (np.sqrt(2)/np.pi**2)*T**(3/2)*FermiDirac.Ionehalf(float(eta))

	@staticmethod	
	@np.vectorize
	def n_free_TF(T, eta, xmid):
		"""
		Current working definition of free vs bound is that particles are K>U, ignoring where the 
		Fermi Level is. That is, x - (eta-mu) > 0, for x the integration variable in I_1/2(eta). 

		(Really probably we should find the bands and in simple liquids define free/valence as the 
		top partially filled band,and all free/valence together would constitute the free density. 
		Only strongly bound, well separated states would contribute to n_b.)
		"""
		# xmid = eta-mu/T
		return (np.sqrt(2)/np.pi**2)*T**(3/2)*FermiDirac.inc_upper_Ionehalf(xmid, eta)

	@staticmethod	
	@np.vectorize
	def n_bound_TF(T, eta, xmid):
		"""
		See above free density for description.
		"""
		# xmid = eta-mu/T
		return (np.sqrt(2)/np.pi**2)*T**(3/2)*FermiDirac.inc_lower_Ionehalf(xmid, eta)

	@staticmethod
	@np.vectorize
	def n_TF_prime(T, eta):
		eps = eta*1e-6
		return (ThomasFermi.n_TF(T, eta+eps)-ThomasFermi.n_TF(T, eta-eps))/(2*eps)

	def make_η(self):
		ηs = np.sort(np.concatenate([np.geomspace(1e-4,1e5,num=1000),-np.geomspace(1e-4,10**(2.5),num=1000)]))
		ρs = self.n_TF(self.T, ηs)
		η_logged = interp1d(np.log(ρs), ηs, kind='linear')#, bounds_error=None, fill_value = [])
		# η = lambda ρ: η_logged(np.log(ρ))
		def η(ρ):
			if ρ<=0:
				return -1e50 #infinity
			elif ρ<1e-8:
				return np.log( 4*ρ*(π/2/self.T)**1.5 )
			elif ρ>1e2:
				return 1/(2*self.T) * (3*π**2*ρ)**(2/3)
			else: 
				return η_logged(np.log(ρ))
		return η

	def η_func(self,ρ):
		if ρ<1e-8:
		    return np.log( 4*ρ*(π/2/self.T)**1.5 )
		elif ρ>1e2:
		    return 1/(2*self.T) * (3*π**2*ρ)**(2/3)
		else:
		    return self.η_interp(ρ)
	
	def Vex_density_simple(cls,rho,*args):
		"""
		Exchange energy density
		"""
		Ce= -0.75 * (3/np.pi)**(1/3)
		Vex = Ce* rho**(1/3)
		return Vex

	def Vc_density_simple(cls,rho,*args):
		"""
		Correlation energy for electron gas
		"""
		if rho==0:
			return 0
		rs=(4/3*np.pi*rho)**(-1/3)
		Ecorr_low = -0.438/rs + 1.325/rs**(3/2) - 1.47/rs**2 - 0.4/rs**(5/2)
		Ecorr_high= 0.0311* np.log(rs) - 0.048 + 0.009*rs*np.log(rs)-0.01*rs
		Ecorr_mid = -(0.06156 - 0.01898* np.log(rs)) # minus sign error in paper!!
		if rs>=10:
			return Ecorr_low
		elif rs>=0.7:
			return Ecorr_mid
		else:
			return Ecorr_high

	def simple_vxc(self, rho):
		return self.Vc_density_simple(rho) + self.Vex_density_simple(rho)

	def Vxc_density(self,rho,*args):
		"""
		Exchange correlation energy density for electron gas
		"""
		libxc_func  = pylibxc.LibXCFunctional("LDA_XC_KSDT",'unpolarized')
		dfdrho  = lambda rho: libxc_func.compute({'rho':rho})['zk'][0,0]
		vdfdrho = np.vectorize(dfdrho)
		return dfdrho(rho)

	def Vxc_density_prime(self,rho,*args):
	    """
	    Full functional derivative of Vxc, delta Vxc/delta n, or if Vxc = integral(vxc), then 
	    this function returns  
	        delta Vxc/delta n = vxc + n d(vxc)/dn
	    """
	    libxc_func  = pylibxc.LibXCFunctional("LDA_XC_GDSMFB",'unpolarized')
	    dfdrho  = lambda rho: libxc_func.compute({'rho':rho})['vrho'][0,0]
	    vdfdrho = np.vectorize(dfdrho)
	    return dfdrho(rho)

	def fast_vxc(self):
	    rhoe = np.geomspace(1e-5,1e8,num=10000)
	    vxc = lambda ρe: self.Vxc_density_prime(ρe) #NO CHAIN RULE, this is done by libxc
	    return np.vectorize(interp1d(rhoe, np.array([vxc(rho) for rho in rhoe]), bounds_error=False, fill_value=(0,None) ))

	# Weizsacker correction stuff
	def fast_hλ(self):
		ηs = np.sort(np.concatenate([np.geomspace(1e-4,1e5,num=1000),-np.geomspace(1e-4,10**(2.5),num=1000)]))
		hλs = self.hλ(ηs)
		hλs = interp1d(ηs, hλs, kind='linear', bounds_error=False, fill_value = (self.hλ(-1e2), self.hλ(1e4)))
		# hλ = lambda η: hλs_logged(np.log(ηs))
		return np.vectorize(hλs)


	@staticmethod
	@np.vectorize
	def hλ(η, λ = 1/9): 
	    return 3*λ/4 * (FermiDirac.Iminusonehalfprime(η) * FermiDirac.Ionehalf(η)/ FermiDirac.Iminusonehalf(η)**2)

	@staticmethod
	@np.vectorize
	def hλ_prime(η, **kwargs): 
	    eps = η*1e-6
	    return (ThomasFermi.hλ(η + eps) - ThomasFermi.hλ(η - eps))/(2*eps)

	def K_correction_coefficients(self, η, ρ, T, second_b_term_zero = 1):
		"""
		Kirzhnits correction, which is basically finite-T Weizsacker.
		Defining coefficients by
		δF_K/δρ = a/ρ nabla^2 (ρ) + b/ρ^2 |nabla(ρ)|^2 
		"""
		a = -2*self.hλ_func(η)
		dηdn = (4*π**2/(2*T)**1.5/FermiDirac.Iminusonehalf(η))
		b = self.hλ_func(η) - second_b_term_zero * ρ * dηdn * self.hλprime_func(η) #Sets second term to zero if second_b_term_zero = 0
		return a, b

	def plot_η_err(self):
		fig, ax = plt.subplots(figsize=(4,3))
		ηs = np.sort(np.concatenate([np.geomspace(1e-3,1e6,num=10000),-np.geomspace(1e-3,1e6,num=10000)]))
		ρs = ThomasFermi.n_TF(self.T, ηs)
		η_approx = np.array([self.η_func(ρ) for ρ in ρs])
		errs = abs(ηs - η_approx)/np.sqrt(ηs**2 + η_approx**2)
		plt.plot(ρs, errs ,'--.')
		ax.set_title(r'$\eta( \rho)$ error', fontsize=10)
		ax.set_ylabel('err', fontsize=10)
		ax.set_xlabel(r'$\rho$', fontsize=10)
		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.set_xlim(1e-10,1e5)
		ax.tick_params(labelsize=10)
	    
