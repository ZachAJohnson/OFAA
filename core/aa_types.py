from scipy.interpolate import interp1d
import numpy as np

import os
from .config import CORE_DIR, PACKAGE_DIR

from .average_atom_new import AverageAtom
from hnc.hnc.constants import *



# Prebuilt Average Atom Models
class AverageAtomFactory:
    @staticmethod
    def create_model(model_type, Z, A, Ti, Te, rs, R, *args, **kwargs):
        if model_type == 'ThomasFermi':
            return ThomasFermiModel(Z, A, Ti, Te, rs, R, *args, **kwargs)
        
        elif model_type == 'NeutralPseudoAtom':
            return NeutralPseudoAtomModel(Z, A, Ti, Te, rs, R,*args, **kwargs)
        
        elif model_type == 'TFStarret2014':
            model_kwargs = {'rmin':1e-3, 'Npoints':1000, 'grid_spacing':'geometric'}
            model_kwargs.update(kwargs)
            return TFStarret2014(Z, A, Ti, Te, rs, R, *args, **model_kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Average Atom Types
class TFStarret2014_EmptyAtom(AverageAtom):
    def __init__(self, Z, A, Ti, Te, rs, R, Zstar,  **kwargs):
        super().__init__(Z, A, Ti, Te, rs, R, Zstar_init=Zstar, **kwargs)
        self.ignore_vxc = kwargs.get('ignore_vxc', True)
        self.aa_type = "Empty"
        # Runs TF to get empty-atom electron density
        self.ne = self.ne_bar*np.ones(self.grid.Nx)
        self.φion = np.zeros_like(self.ne)
        self.Qion = self.grid.integrate_f( self.ρi )
        self.ne_bar = self.Zstar*self.ni_bar
        self.μ = self.get_μ_infinite()

    def update_bulk_params(self):
        pass

    def get_ne_guess(self):
        ne_guess = self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar)
        return ne_guess

    def get_βVeff(self, φe, ne, ne_bar):
        βVeff = ( -φe - self.φion + self.vxc_f(ne) - self.vxc_f(ne_bar) )/self.Te
        return βVeff

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

    def update_bulk_params(self):    
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

    def get_βVeff(self, φe, ne, ne_bar):
        βVeff = ( -φe - self.φion + self.vxc_f(ne) - self.vxc_f(ne_bar) )/self.Te
        return βVeff

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

    def setup_core_atom(self):
        self.core_atom = TFStarret2014_core(self.Z, self.A, self.Ti, self.Te, self.rs, **self.kwargs)

    def setup_empty_atom(self):
        self.empty_atom = TFStarret2014_EmptyAtom(self.Z, self.A, self.Ti, self.Te, self.rs, self.R, self.core_atom.Zstar, **self.kwargs)

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        print("Setting up and solving core.")
        self.setup_core_atom()
        self.core_atom.solve(**kwargs)
        print("Settin up and solving empty atom")
        self.setup_empty_atom()
        self.empty_atom.solve(**kwargs)
        self.combine_models()
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
         
    ### Saving data
    def save_data(self):
        # Electron File
        err_info = " " #f"# Convergence: Err(φ)={self.poisson_err:.3e}, Err(n_e)={self.rho_err:.3e}, Err(IET)={self.iet.final_Picard_err:.3e}, Q_net={self.Q:.3e}\n"
        aa_info = '# {{"name":"{0}", "Z":{1}, "Zstar":{2}, "A":{3},  "μ_core[AU]": {4:.10e}, "μ_empty[AU]": {5:.10e}, "Te[AU]": {6:.3e}, "Ti[AU]": {7:.3e}, "rs[AU]": {8:.3e} }}\n'.format("CS2014_" + self.name, self.Z, self.Zstar, self.A, self.core_atom.μ, self.empty_atom.μ, self.Te,self.Ti, self.rs)
        column_names = f"   {'r[AU]':15} {'ne_full[AU]':15} {'n_ion[AU]':15} {'ne_ext[AU]':15} {'ne_PA[AU]':15} {'ne_scr[AU]':15} "
        header = ("Model replicates Starrett & Saumon 2014 'A simple method for determining the ionic structure of warm dense matter'\n" + 
        #           "# All units in Hartree [AU] if not specified\n"+
                err_info + aa_info + column_names)   
        data = np.array([self.grid.xs, self.ne_full, self.n_ion, self.ne_ext, self.ne_PA, self.ne_scr] ).T
        
        txt='{0}_{1}_R{2:.1e}_rs{3:.1e}_Te{4:.1e}eV_Ti{5:.1e}eV_electron_info.dat'.format(self.name, self.aa_type, self.R, self.rs, self.Te*AU_to_eV, self.Ti*AU_to_eV, self.Zstar)
        self.savefile = os.path.join(PACKAGE_DIR,"data",txt)
        np.savetxt(self.savefile, data, delimiter = ' ', header=header, fmt='%15.6e', comments='')

        # # Ion file
        # err_info = f"# Convergence: Err(φ)={self.poisson_err:.3e}, Err(n_e)={self.rho_err:.3e}, Err(IET)={self.iet.final_Picard_err:.3e}, Q_net={self.Q:.3e}\n"
        # aa_info = '# {{"name":"{0}", "Z":{1}, "Zstar":{2}, "A":{3},  "μ[AU]": {4:.10e}, "Te[AU]": {5:.3e}, "Ti[AU]": {6:.3e}, "rs[AU]": {7:.3e} }}\n'.format(self.name, self.Z, self.Zstar, self.A, self.μ, self.Te, self.Ti, self.rs)
        # column_names = f"   {'r[AU]':15} {'U_ei[AU]':15} {'U_ii[AU]':15} {'g_ii':15} "
        # header = ("Model replicates Starrett & Saumon 2014 'A simple method for determining the ionic structure of warm dense matter'\n" + 
        #           "# All units in Hartree [AU] if not specified\n"+
        #             err_info + aa_info + column_names)   
        # data = np.array([self.iet.r_array*self.rs, self.Uei_iet, self.iet.βu_r_matrix[0,0]*self.Ti , self.iet.h_r_matrix[0,0]+1 ] ).T
        
        # txt='{0}_{1}_R{2:.1e}_rs{3:.1e}_Te{4:.1e}eV_Ti{5:.1e}eV_IET_info.dat'.format(self.name, self.aa_type, self.R, self.rs, self.Te*AU_to_eV, self.Ti*AU_to_eV, self.Zstar)
        # self.savefile = os.path.join(PACKAGE_DIR,"data",txt)
        # np.savetxt(self.savefile, data, delimiter = ' ', header=header, fmt='%15.6e', comments='')

class ThomasFermiModel(AverageAtom):
    def __init__(self, Z, A, Ti, Te, rs, R, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, R, **kwargs)
        self.ignore_vxc = kwargs.get('ignore_vxc', True)
        self.aa_type = "TF"

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        self.solve_TF(**kwargs)

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


