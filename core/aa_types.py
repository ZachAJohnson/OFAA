from scipy.interpolate import interp1d
import numpy as np

import os
from .config import CORE_DIR, PACKAGE_DIR

from .average_atom_new import AverageAtom
from .solvers import jacobi_relaxation, sor, gmres_ilu, tridiagsolve, Ndiagsolve
from hnc.hnc.constants import *

from average_atom.core.physics import  FermiDirac

# Prebuilt Average Atom Models
class AverageAtomFactory:
    @staticmethod
    def create_model(model_type, Z, A, Ti, Te, rs, R, *args, **kwargs):
        if model_type == 'ZJ_ISModel':
            model_kwargs = {'rmin':1e-3, 'Npoints':1000, 'grid_spacing':'geometric','iet_R_over_rs':10}
            model_kwargs.update(kwargs)
            return ZJ_ISModel(Z, A, Ti, Te, rs, R, *args, **model_kwargs)
        
        elif model_type == 'ZJ_ISModel_W':
            model_kwargs = {'Weizsacker_λ':1/5, 'rmin':1e-3, 'Npoints':1000, 'grid_spacing':'geometric','iet_R_over_rs':10}
            model_kwargs.update(kwargs)
            return ZJ_ISModel_W(Z, A, Ti, Te, rs, R, *args, **model_kwargs)

        elif model_type == 'ZJ_CSModel':
            model_kwargs = {'rmin':1e-3, 'Npoints':1000, 'grid_spacing':'geometric'}
            model_kwargs.update(kwargs)
            return ZJ_CSModel(Z, A, Ti, Te, rs, R, *args, **model_kwargs)
        
        elif model_type == 'NeutralPseudoAtom':
            return NeutralPseudoAtomModel(Z, A, Ti, Te, rs, R,*args, **kwargs)
        
        elif model_type == 'TFStarret2014':
            model_kwargs = {'rmin':1e-3, 'Npoints':1000, 'grid_spacing':'geometric','χ_type':'TF', 
                            'iet_R_over_rs' : 20, 'iet_N_bins': 2000}
            model_kwargs.update(kwargs)
            return TFStarret2014(Z, A, Ti, Te, rs, R, *args, **model_kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Average Atom Types
class TFStarret2014_EmptyAtom(AverageAtom):
    def __init__(self, Z, A, Ti, Te, rs, R, Zstar, μ,  **kwargs):
        super().__init__(Z, A, Ti, Te, rs, R, Zstar_init=Zstar, **kwargs)
        self.ignore_vxc = kwargs.get('ignore_vxc', True)
        self.aa_type = "Empty"
        # Runs TF to get empty-atom electron density
        self.ne = self.ne_bar*np.ones(self.grid.Nx)
        self.φion = np.zeros_like(self.ne)
        self.Qion = self.grid.integrate_f( self.ρi )
        self.ne_bar = self.Zstar*self.ni_bar
        self.μ = μ

    def update_bulk_params(self, iters):
        pass

    def get_ne_guess(self):
        ne_guess = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)
        return ne_guess

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        model_kwargs = {'picard_alpha':0.5, 'tol':1e-10}
        model_kwargs.update(kwargs)
        self.solve_TF(**model_kwargs)

class TFStarret2014_core(AverageAtom):
    """
    Based on Charlie Starrett's (CS) and Didier Saumon (2014)
        'A simple method for determining the ionic structure of warm dense matter' 
    """
    def __init__(self, Z, A, Ti, Te, rs, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, rs, **kwargs)
        self.ignore_vxc = kwargs.get('ignore_vxc', True)
        self.aa_type = "CS2014"
        self.name=""

    def update_bulk_params(self, iters):    
        # self.update_newton_Zstar()          # get Zstar from bound/free
        # if n%10==0 or n<5 and not μ_converged:
        self.set_μ_neutral()
        self.ne_bar = self.ne[-1] # a CS specific idea
        self.Zstar = self.ne_bar/self.ni_bar

        # elif not μ_converged: 
            # self.update_μ_newton(alpha1=1e-3)
        
    def get_ne_guess(self):
        ne_guess = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)
        return ne_guess

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        model_kwargs = {'picard_alpha':0.5, 'tol':1e-10}
        model_kwargs.update(kwargs)
        self.solve_TF(**model_kwargs)
        self.set_physical_params()
        self.make_bound_free()
        self.ne_bar = self.ne[-1] # a CS specific idea
        self.Zstar = self.ne_bar/self.ni_bar

class TFStarret2014(AverageAtom):
    """
    Based on Charlie Starrett's (CS) and Didier Saumon (2014)
        'A simple method for determining the ionic structure of warm dense matter' 
    """
    def __init__(self, Z, A, Ti, Te, rs, R, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, R, initialize=False, **kwargs)
        self.ignore_vxc = kwargs.get('ignore_vxc', True)
        self.aa_type = "CS2014"
        self.kwargs = kwargs
        self.setup_core_atom()

    def setup_core_atom(self):
        self.core_atom = TFStarret2014_core(self.Z, self.A, self.Ti, self.Te, self.rs, **self.kwargs)

    def setup_empty_atom(self):
        self.empty_atom = TFStarret2014_EmptyAtom(self.Z, self.A, self.Ti, self.Te, self.rs, self.R, self.core_atom.Zstar, self.core_atom.μ, **self.kwargs)

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        print("Solving core.")
        self.core_atom.solve(**kwargs)
        print("Settin up and solving empty atom")
        self.setup_empty_atom()
        self.empty_atom.solve(**kwargs)
        self.combine_models()
        self.μ = self.core_atom.μ
        self.make_iet()
        self.set_uii_eff()
        self.save_data()

    def combine_models(self):
        logr_data = np.log(self.core_atom.grid.xs)
        log_ne_data = np.where(self.core_atom.ne==0, np.log(1e-20), np.log(self.core_atom.ne) )
        log_nb_data = np.where(self.core_atom.nb==0, np.log(1e-20), np.log(self.core_atom.nb) )
        
        self.ne_full = np.exp(interp1d(logr_data, log_ne_data, bounds_error=False, fill_value = (log_ne_data[0], log_ne_data[-1]) )(np.log(self.grid.xs)))
        self.n_ion   = np.exp(interp1d(logr_data, log_nb_data, bounds_error=False, fill_value = (log_nb_data[0], log_nb_data[-1]) )(np.log(self.grid.xs)))
        self.ne_ext  = self.empty_atom.ne
        self.ne_PA   = self.ne_full - self.ne_ext
        self.ne_scr  = self.ne_PA - self.n_ion
        self.ρi = self.empty_atom.ρi
        self.Zstar = self.grid.integrate_f(self.ne_scr)
        print(f"Check combination is neutral: {self.Z} = {self.grid.integrate_f(self.ne_PA):0.3e}")
    
    def make_Uei_iet(self):
        self.ne_scr_r  = interp1d(self.grid.xs, self.ne_scr , bounds_error=False, fill_value='extrapolate')(self.iet.r_array*self.rs)        
        self.ne_scr_k  = self.rs**3 * self.iet.FT_r_2_k(self.ne_scr_r)
        self.Uei_iet_k = self.ne_scr_k/self.χee_iet
        self.Uei_iet   = self.iet.FT_k_2_r(self.Uei_iet_k**self.rs**-3) 
     
    ### Saving data
    def save_data(self):
        # Electron File
        err_info = f"# Convergence: Err(φ)={self.core_atom.poisson_err:.3e}, Err(n_e)={self.core_atom.rho_err:.3e}, Err(IET)={self.iet.final_Picard_err:.3e}, Q_net={self.core_atom.Q:.3e}\n"
        aa_info = '# {{"name":"{0}", "Z":{1}, "Zstar":{2}, "A":{3},  "μ[AU]": {4:.10e}, "Te[AU]": {5:.3e}, "Ti[AU]": {6:.3e}, "rs[AU]": {7:.3e} }}\n'.format("CS2014_" + self.name, self.Z, self.Zstar, self.A, self.core_atom.μ, self.Te,self.Ti, self.rs)
        column_names = f"   {'r[AU]':15} {'ne_full[AU]':15} {'n_ion[AU]':15} {'ne_ext[AU]':15} {'ne_PA[AU]':15} {'ne_scr[AU]':15} "
        header = ("# Model replicates Starrett & Saumon 2014 'A simple method for determining the ionic structure of warm dense matter'\n" + 
        #           "# All units in Hartree [AU] if not specified\n"+
                err_info + aa_info + column_names)   
        data = np.array([self.grid.xs, self.ne_full, self.n_ion, self.ne_ext, self.ne_PA, self.ne_scr] ).T
        
        txt='{0}_{1}_R{2:.1e}_rs{3:.1e}_Te{4:.1e}eV_Ti{5:.1e}eV_electron_info.dat'.format(self.name, self.aa_type, self.R, self.rs, self.Te*AU_to_eV, self.Ti*AU_to_eV, self.Zstar)
        self.savefile   = os.path.join(self.savefolder,txt)
        np.savetxt(self.savefile, data, delimiter = ' ', header=header, fmt='%15.6e', comments='')

        # Ion file
        err_info = f"# Convergence: Err(φ)={self.core_atom.poisson_err:.3e}, Err(n_e)={self.core_atom.rho_err:.3e}, Err(IET)={self.iet.final_Picard_err:.3e}, Q_net={self.core_atom.Q:.3e}\n"
        aa_info = '# {{"name":"{0}", "Z":{1}, "Zstar":{2}, "A":{3},  "μ[AU]": {4:.10e}, "Te[AU]": {5:.3e}, "Ti[AU]": {6:.3e}, "rs[AU]": {7:.3e} }}\n'.format(self.name, self.Z, self.Zstar, self.A, self.core_atom.μ, self.Te, self.Ti, self.rs)
        column_names = f"   {'r[AU]':15} {'U_ei[AU]':15} {'U_ii[AU]':15} {'g_ii':15} "
        header = ("# Model replicates Starrett & Saumon 2014 'A simple method for determining the ionic structure of warm dense matter'\n" + 
                  "# All units in Hartree [AU] if not specified\n"+
                    err_info + aa_info + column_names)   
        data = np.array([self.iet.r_array*self.rs, self.Uei_iet, self.iet.βu_r_matrix[0,0]*self.Ti , self.iet.h_r_matrix[0,0]+1 ] ).T
        
        txt='{0}_{1}_R{2:.1e}_rs{3:.1e}_Te{4:.1e}eV_Ti{5:.1e}eV_IET_info.dat'.format(self.name, self.aa_type, self.R, self.rs, self.Te*AU_to_eV, self.Ti*AU_to_eV, self.Zstar)
        self.savefile = os.path.join(self.savefolder,txt)
        np.savetxt(self.savefile, data, delimiter = ' ', header=header, fmt='%15.6e', comments='')

class ZJ_ISModel(AverageAtom):
    """
    Ion sphere model with R=rs
    """
    def __init__(self, Z, A, Ti, Te, rs, R_extended, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, rs, **kwargs)
        self.R_extended = R_extended
        self.ignore_vxc = kwargs.get('ignore_vxc', True)
        self.aa_type = "TF"

    def update_bulk_params(self, iters):    
        # if iters >= 50:
        #     # old_ne_bar = self.ne_bar
        #     # if iters%25==0:
        #     #     self.update_newton_Zstar()
        #     # # self.ne   += self.ne_bar - old_ne_bar
        if iters%10==0:
            self.set_μ_neutral()
        self.ne_bar = self.ne[-1]
        # else:
        #     self.update_μ_newton(alpha1=1e-3)
        # self.set_μ_neutral()
        # self.update_μ_newton(alpha1=1e-3)


    def get_ne_guess(self):
        # ne_guess = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)
        ne_guess = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne[-1])
        return ne_guess

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        self.solve_kwargs = {'picard_alpha':0.5, 'tol':1e-10}
        self.solve_kwargs.update(kwargs)
        self.solve_TF(**self.solve_kwargs)
        self.set_physical_params()
        self.make_bound_free()
        self.set_Zstar()
        self.set_uii_eff()

    def get_E(self):
        βVeff = self.get_βVeff(self.φe, self.ne, self.ne_bar)
        η = self.μ/self.Te - βVeff
        βU = self.grid.integrate_f(βVeff*self.ne)

        φ_from_e_only = self.get_φe(-self.ne)[0] 
        φ_from_i_only = self.get_φe(self.ρi)[0] + self.φion 
        U = -self.grid.integrate_f(self.ne * (0.5*φ_from_e_only + φ_from_i_only))

        I_onehalf = FermiDirac.Ionehalf(η)
        I_threehalf = FermiDirac.Ithreehalf(η)

        K = (2*self.Te)**1.5/(2*π**2) * self.Te * self.grid.integrate_f( I_threehalf)
        self.Ke = K
        self.Ue = U
        self.Ee = U+K
        return U, K, (U + K)

    def get_P(self):
        self.Pe = (2*self.Te)**2.5/(6*π**2) * FermiDirac.Ithreehalf(self.μ/self.Te)
        return self.Pe

    def print_EOS(self):
        Ee_pot, Ee_K, Ee = self.get_E()
        P_e =  self.get_P()
        print(f"Ee_pot_density = {Ee_pot*self.ni_bar:0.3e} [au], Ee_K = {Ee_K*self.ni_bar:0.3e} [au], Ee_tot = {Ee*self.ni_bar:0.3e} [au]")
        print(f"P_e = {P_e:0.3e} [au], {P_e*AU_to_bar/1e6:0.3e} [Mbar]")
        print(f"Virial if {Ee_K:0.3e} = {3/2*P_e*self.Vol - 0.5*Ee_pot:0.3e} --->  off by {100*Ee_K/(3/2*P_e*self.Vol - 0.5*Ee_pot) - 100:0.3e} % ")

        print(f"\nβEe_pot/Z = {Ee_pot/self.Te/self.Z:0.3f}, βEe_K/Z = {Ee_K/self.Te/self.Z:0.3f}, βEe_tot/Z = {Ee/self.Te/self.Z:0.3f}")
        print(f"βP_e Ω/Z = {P_e*self.Vol/self.Te/self.Z:0.3f}")
        # print(f"Virial if {Ee_K:0.3e} = {3/2*P_e*self.Vol - 0.5*Ee_pot:0.3e}")


class ZJ_CSModel(AverageAtom): 
    """
    Correlation sphere model with R>>rs
    """

    def __init__(self, Z, A, Ti, Te, rs, R, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, R, **kwargs)
        self.ignore_vxc = kwargs.get('ignore_vxc', True)
        self.aa_type = "TF"

    def update_bulk_params(self, iters):    
        if iters > 50:
            old_ne_bar = self.ne_bar
            self.update_newton_Zstar()
            self.ne   += self.ne_bar - old_ne_bar
        self.update_ρi_and_Zstar_to_make_neutral()
        self.set_μ_infinite()
        self.set_μ_neutral()

    def get_ne_guess(self):
        ne_guess = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)
        return ne_guess

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        self.solve_kwargs = {'picard_alpha':0.5, 'tol':1e-10}
        self.solve_kwargs.update(kwargs)
        self.solve_TF(**self.solve_kwargs)
        self.set_physical_params()
        self.make_bound_free()
        self.set_uii_eff()


class ZJ_ISModel_W(AverageAtom):
    """
    Ion sphere model with R=rs
    """
    def __init__(self, Z, A, Ti, Te, rs, R_extended, Weizsacker_λ=1/5, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, rs, **kwargs)
        self.R_extended = R_extended
        self.ignore_vxc = kwargs.get('ignore_vxc', True)

        self.aa_type = "TFW"

        # Gradient Corrections
        self.gradient_correction = True
        self.set_gradient_correction()
        self.λ_W = Weizsacker_λ


    def update_bulk_params(self, iters):    
        # if iters >= 50:
        #     # old_ne_bar = self.ne_bar
        #     # if iters%25==0:
        #     #     self.update_newton_Zstar()
        #     # # self.ne   += self.ne_bar - old_ne_bar
        if iters%10==0:
            self.set_μ_neutral()
        self.ne_bar = self.ne[-1]
        # else:
        #     self.update_μ_newton(alpha1=1e-3)
        # self.set_μ_neutral()
        # self.update_μ_newton(alpha1=1e-3)


    def get_ne_guess(self):
        ne_guess = self.get_ne_W()
        return ne_guess

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        self.solve_kwargs = {'picard_alpha':0.1, 'tol':1e-10}
        self.solve_kwargs.update(kwargs)
        self.solve_TF(**self.solve_kwargs)
        self.set_physical_params()
        self.make_bound_free()
        self.set_Zstar()
        self.set_uii_eff()


    def set_gradient_correction(self):
        """
        Sets the coefficients and overall gradient correction energy
        [1] - Mod-MD Murillo paper <https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021044>
        Args:
            str gradient_correction_type: 'K' is Kirzhnits, 'W' is Weizsacker for arbitrary λ

        """
        # if self.gradient_correction=='K':
        #     K_coeff_func = lambda *args: self.TF.K_correction_coefficients(*args, second_b_term_zero=1)
        #     self.get_grad_coeffs = np.vectorize(K_coeff_func)
        # elif self.gradient_correction=='W':
        self.get_grad_coeffs = lambda *args: [ -1/4*self.λ_W, 1/8*self.λ_W ]
        self.get_grad_coeffs = np.vectorize(self.get_grad_coeffs)

    def get_gradient_energy(self, ne):
        """
        Defined coefficients by
        δF_K/δρ = a/ρ nabla^2 (ρ) + b/ρ^2 |nabla(ρ)|^2 
        """
        grad_n = self.grid.dfdx(ne)
        laplace_ne = self.grid.A_laplace.dot(ne)

        aW, bW = self.get_grad_coeffs()
        return aW/ne * laplace_ne + bW/ne**2 * grad_n**2
 

    def get_eta_from_sum(self, φe, ne, μ, ne_bar):
        βVeff = self.get_βVeff(φe, ne, ne_bar)
        eta = μ/self.Te - βVeff - self.get_gradient_energy(ne)/self.Te
        return eta

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
        ne = self.fast_n_TF( eta )
        return ne

    def get_ne_W(self, constant_hλ=False):

        def banded_Ab():
            # -1/2 d^2/dr2 γ  + 1/λ veff γ = μ/λ γ
            # -1/2 d^2/dr2 γ  = 1/λ (μ - veff)  γ, now use  x = min(x,0) + max(x,0)
            # -1/2 d^2/dr2 γ - 1/λ min[0, (μ - veff)]  γ  = 1/λ max[0, (μ - veff)]  γ 


            # bc_vec(γ) = 2/λW (μ - ηT - Vcol - Vxc ) 
            η = self.TF.η_interp(self.ne)
            βVeff = self.get_βVeff(self.φe, self.ne, self.ne_bar)
            rhs_vec  = 1/self.λ_W*(self.μ - self.Te*βVeff - self.Te*η )
            
            # Split to gaurantee positive solution 
            γ =  np.sqrt(self.ne)*self.grid.xs
            b =  np.max([0*self.grid.xs, rhs_vec], axis=0)*γ # will go on right side
            c = -np.min([0*self.grid.xs, rhs_vec], axis=0) # will go on left side
    
            b = 1/self.λ_W*self.μ*γ
            c = 1/self.λ_W*(self.Te*βVeff + self.Te*η )

            A = -1/2 * self.grid.A_laplace + np.diag(c)

            #Boundary
            dx= self.grid.dx
            x = self.grid.xs
            
            # Interior- Kato cusp condition
            interior_bc = 'zero'
            if interior_bc in ['kato','Kato']:
                # A[0,0]  = 1; A[0,1] = -1/(1 + self.grid.dx[0]*(-self.Z+ 1/self.grid.xs[1])) 
                # A[0,0:2] *= 1/self.grid.dx[0]**2  #get number more similar to other matrix values
                # b[0] = -(self.Z + 1/self.grid.xs[0])*γ[0]/self.grid.dx[0]
                
                # retry- use Z~0 for rmin<<1, so exactly γ(r)=r/r_0 γ(r_0), applied to first point, or γ[1] - γ[0]x[1]/x[0]=0
                A[0, :] = self.grid.A_dfdx[0,:] 
                b[0]   = γ[0]*(1/x[0] - self.Z)
                

            elif interior_bc == 'zero': # Sets γ to zero
                A[0,0] = 1/self.grid.dx[0]**2
                b[0] = 0

            # Exterior- γ''= 0 at edge is nice
            # A[-1,-1] =  2/dx[-1]  /(dx[-1] + dx[-2])
            # A[-1,-2] = -2/(dx[-1]*dx[-2])           
            # A[-1,-3] =  2/dx[-2]/(dx[-1] + dx[-2])
            
            # Exterior- dγdr=γ/r
            A[-1,:] = self.grid.A_dfdx[-1,:]

            b[-1]  = γ[-1]/x[-1]
            
            return A, b     

        A, b = banded_Ab()
        self.Ab_W = A, b
        γs = Ndiagsolve(A, b, self.N_stencil_oneside)
        self.γs = np.max([γs, np.zeros_like(self.ne)], axis=0)

        new_ne = γs**2/self.grid.xs**2
        new_ne = new_ne * self.Z/self.grid.integrate_f(new_ne) # normalize so it is reasonable

        return new_ne

    def get_E(self):
        βVeff = self.get_βVeff(self.φe, self.ne, self.ne_bar)
        η = self.μ/self.Te - βVeff
        βU = self.grid.integrate_f(βVeff*self.ne)

        φ_from_e_only = self.get_φe(-self.ne)[0] 
        φ_from_i_only = self.get_φe(self.ρi)[0] + self.φion 
        U = -self.grid.integrate_f(self.ne * (0.5*φ_from_e_only + φ_from_i_only))

        I_onehalf = FermiDirac.Ionehalf(η)
        I_threehalf = FermiDirac.Ithreehalf(η)

        K = (2*self.Te)**1.5/(2*π**2) * self.Te * self.grid.integrate_f( I_threehalf)
        self.Ke = K
        self.Ue = U
        self.Ee = U+K
        return U, K, (U + K)

    def get_P(self):
        self.Pe = (2*self.Te)**2.5/(6*π**2) * FermiDirac.Ithreehalf(self.μ/self.Te)
        return self.Pe

    def print_EOS(self):
        Ee_pot, Ee_K, Ee = self.get_E()
        P_e =  self.get_P()
        print(f"Ee_pot_density = {Ee_pot*self.ni_bar:0.3e} [au], Ee_K = {Ee_K*self.ni_bar:0.3e} [au], Ee_tot = {Ee*self.ni_bar:0.3e} [au]")
        print(f"P_e = {P_e:0.3e} [au], {P_e*AU_to_bar/1e6:0.3e} [Mbar]")
        print(f"Virial if {Ee_K:0.3e} = {3/2*P_e*self.Vol - 0.5*Ee_pot:0.3e} --->  off by {100*Ee_K/(3/2*P_e*self.Vol - 0.5*Ee_pot) - 100:0.3e} % ")

        print(f"\nβEe_pot/Z = {Ee_pot/self.Te/self.Z:0.3f}, βEe_K/Z = {Ee_K/self.Te/self.Z:0.3f}, βEe_tot/Z = {Ee/self.Te/self.Z:0.3f}")
        print(f"βP_e Ω/Z = {P_e*self.Vol/self.Te/self.Z:0.3f}")
        # print(f"Virial if {Ee_K:0.3e} = {3/2*P_e*self.Vol - 0.5*Ee_pot:0.3e}")


class TFW_Model(AverageAtom):
    def __init__(self, Z, A, Ti, Te, rs, R, Weizsacker_λ = 1/5, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, R, **kwargs)
        self.aa_type = "TFW"

        # Gradient Corrections
        self.gradient_correction = True
        self.set_gradient_correction()
        self.λ_W = Weizsacker_λ

    def set_gradient_correction(self):
        """
        Sets the coefficients and overall gradient correction energy
        [1] - Mod-MD Murillo paper <https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021044>
        Args:
            str gradient_correction_type: 'K' is Kirzhnits, 'W' is Weizsacker for arbitrary λ

        """
        # if self.gradient_correction=='K':
        #     K_coeff_func = lambda *args: self.TF.K_correction_coefficients(*args, second_b_term_zero=1)
        #     self.get_grad_coeffs = np.vectorize(K_coeff_func)
        # elif self.gradient_correction=='W':
        self.get_grad_coeffs = lambda *args: [ -1/4*self.λ_W, 1/8*self.λ_W ]
        self.get_grad_coeffs = np.vectorize(self.get_grad_coeffs)

    def get_gradient_energy(self, ne):
        """
        Defined coefficients by
        δF_K/δρ = a/ρ nabla^2 (ρ) + b/ρ^2 |nabla(ρ)|^2 
        """
        grad_n = self.grid.dfdx(ne)
        laplace_ne = self.grid.A_laplace.dot(ne)

        if self.gradient_correction=='K':
            eta = self.TF.η_interp(ne)
            aW, bW = self.get_grad_coeffs(eta, ne, self.T)
        elif self.gradient_correction=='W':
            aW, bW = self.get_grad_coeffs()
        return aW/ne * laplace_ne + bW/ne**2 * grad_n**2
    
    # def get_ne_W(self, constant_hλ=False):

    #   def banded_Ab():
    #       etas = self.TF.η_interp(self.ne)
    #       if self.gradient_correction=='K':
    #           aW, bW = self.get_grad_coeffs(etas, self.ne, self.Te)
    #           print("bW term NOT supported, please switch to 'W' gradient_correction_type for now.")
    #       elif self.gradient_correction=='W':
    #           aW, bW = self.get_grad_coeffs()

    #       if not self.ignore_vxc:
    #           bc_vec  = -1/(2*aW)*(self.Te*etas - (self.μ + self.ϕe + self.ϕion - self.vxc_f(self.ne) + self.vxc_f(self.ne_bar)) )
    #       if self.ignore_vxc:
    #           bc_vec  = -1/(2*aW)*(self.Te*etas - (self.μ + self.ϕe + self.ϕion ))
                    
    #       # Split to gaurantee positive solution 
    #       γ = np.sqrt(self.ne)*self.grid.xs
    #       b = np.max([0*self.grid.xs, -bc_vec], axis=0)*γ # will go on right side
    #       c = np.max([0*self.grid.xs,  bc_vec], axis=0) # will go on left side

    #       A = -self.grid.A_d2fdx2 + np.diag(c)

    #       #Boundary
    #       dx= self.grid.dx
    #       x = self.grid.xs
            
    #       # Interior- Kato cusp condition
    #       interior_bc = 'kato'
    #       if interior_bc in ['kato','Kato']:
    #           # A[0,0]  = 1; A[0,1] = -1/(1 + self.grid.dx[0]*(-self.Z+ 1/self.grid.xs[1])) 
    #           # A[0,0:2] *= 1/self.grid.dx[0]**2  #get number more similar to other matrix values
    #           # b[0] = -(self.Z + 1/self.grid.xs[0])*γ[0]/self.grid.dx[0]
                
    #           # retry- use Z~0 for rmin<<1, so exactly γ(r)=r/r_0 γ(r_0), applied to first point, or γ[1] - γ[0]x[1]/x[0]=0
    #           A[0, :] = self.grid.A_dfdx[0,:] 
    #           b[0]   = γ[0]*(1/x[0] - self.Z)
                

    #       elif interior_bc == 'zero': # Sets γ to zero
    #           A[0,0] = 1/self.grid.dx[0]**2
    #           b[0] = 0

    #       # Exterior- γ''= 0 at edge is nice
    #       # A[-1,-1] =  2/dx[-1]  /(dx[-1] + dx[-2])
    #       # A[-1,-2] = -2/(dx[-1]*dx[-2])           
    #       # A[-1,-3] =  2/dx[-2]/(dx[-1] + dx[-2])
            
    #       # Exterior- dγdr=γ/r
    #       A[-1,:] = self.grid.A_dfdx[-1,:]

    #       b[-1]  = γ[-1]/x[-1]
            

    #       return A, b     

    #   A, b = banded_Ab()
    #   self.Ab_W = A, b
    #   γs = Ndiagsolve(A, b, self.N_stencil_oneside)
    #   self.γs = γs

    #   new_ne = γs**2/self.grid.xs**2
    #   new_ne = new_ne * self.Z/self.grid.integrate_f(new_ne) # normalize so it is reasonable

    #   return new_ne
    def get_ne_W(self, constant_hλ=False):

        def banded_Ab():
            # - d^2/dr2 γ = bc_vec(γ) γ 
            # bc_vec(γ) = 2/λW (μ - ηT - Vcol - Vxc ) 
            etas = self.TF.η_interp(self.ne)
            if not self.ignore_vxc:
                bc_vec  = 2/self.λ_W*(-self.Te*etas + (self.μ - self.ϕe - self.ϕion + self.vxc_f(self.ne) - self.vxc_f(self.ne_bar)) )
            if self.ignore_vxc:
                bc_vec  = 2/self.λ_W*(-self.Te*etas + (self.μ - self.ϕe - self.ϕion ))
                    
            # Split to gaurantee positive solution 
            γ = np.sqrt(self.ne)*self.grid.xs
            b = -np.max([0*self.grid.xs, -bc_vec], axis=0)*γ # will go on right side
            c = -np.max([0*self.grid.xs,  bc_vec], axis=0) # will go on left side

            A = -self.grid.A_d2fdx2 + np.diag(c)

            #Boundary
            dx= self.grid.dx
            x = self.grid.xs
            
            # Interior- Kato cusp condition
            interior_bc = 'kato'
            if interior_bc in ['kato','Kato']:
                # A[0,0]  = 1; A[0,1] = -1/(1 + self.grid.dx[0]*(-self.Z+ 1/self.grid.xs[1])) 
                # A[0,0:2] *= 1/self.grid.dx[0]**2  #get number more similar to other matrix values
                # b[0] = -(self.Z + 1/self.grid.xs[0])*γ[0]/self.grid.dx[0]
                
                # retry- use Z~0 for rmin<<1, so exactly γ(r)=r/r_0 γ(r_0), applied to first point, or γ[1] - γ[0]x[1]/x[0]=0
                A[0, :] = self.grid.A_dfdx[0,:] 
                b[0]   = γ[0]*(1/x[0] - self.Z)
                

            elif interior_bc == 'zero': # Sets γ to zero
                A[0,0] = 1/self.grid.dx[0]**2
                b[0] = 0

            # Exterior- γ''= 0 at edge is nice
            # A[-1,-1] =  2/dx[-1]  /(dx[-1] + dx[-2])
            # A[-1,-2] = -2/(dx[-1]*dx[-2])           
            # A[-1,-3] =  2/dx[-2]/(dx[-1] + dx[-2])
            
            # Exterior- dγdr=γ/r
            A[-1,:] = self.grid.A_dfdx[-1,:]

            b[-1]  = γ[-1]/x[-1]
            
            return A, b     

        A, b = banded_Ab()
        self.Ab_W = A, b
        γs = Ndiagsolve(A, b, self.N_stencil_oneside)
        self.γs = γs

        new_ne = γs**2/self.grid.xs**2
        new_ne = new_ne * self.Z/self.grid.integrate_f(new_ne) # normalize so it is reasonable

        return new_ne

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        self.solve_TF(**kwargs)

class NeutralPseudoAtomModel(AverageAtom):
    def __init__(self, Z, A, Ti, Te, rs, R, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, R, **kwargs)
        self.aa_type = "NPA"

def solve(self, **kwargs):
    # Neutral Pseudo Atom specific solving logic
    self.solve_TF(**kwargs)


