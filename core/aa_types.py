from .average_atom_new import AverageAtom

# Prebuilt Average Atom Models
class AverageAtomFactory:
    @staticmethod
    def create_model(model_type, Z, A, Ti, Te, rs, R, **kwargs):
        if model_type == 'ThomasFermi':
            return ThomasFermiModel(Z, A, Ti, Te, rs, R, **kwargs)
        elif model_type == 'NeutralPseudoAtom':
            return NeutralPseudoAtomModel(Z, A, Ti, Te, rs, R, **kwargs)
        elif model_type == 'TFStarret2014':
            return TFStarret2014(Z, A, Ti, Te, rs, R, **kwargs)
        elif model_type == 'EmptyAtom':
            return EmptyAtom(Z, A, Ti, Te, rs, R, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Average Atom Types
class EmptyAtom(AverageAtom):
    def __init__(self, Z, A, Ti, Te, rs, R, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, R, **kwargs)
        self.ignore_vxc = kwargs.get('ignore_vxc', True)
        self.aa_type = "Empty"

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        self.solve_TF(**kwargs)

class TFStarret2014(AverageAtom):
    def __init__(self, Z, A, Ti, Te, rs, R, **kwargs):
        super().__init__(Z, A, Ti, Te, rs, R, **kwargs)
        self.ignore_vxc = kwargs.get('ignore_vxc', True)
        self.aa_type = "CS2014"

    def solve(self, **kwargs):
        # Thomas-Fermi specific solving logic
        self.solve_TF(**kwargs)

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


