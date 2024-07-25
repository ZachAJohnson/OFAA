import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres, LinearOperator, eigs, spsolve
from scipy.linalg import solve_banded
from scipy.optimize import root, newton_krylov, fsolve
from scipy.special import gammaincc, gamma

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from .config import CORE_DIR, PACKAGE_DIR
from hnc.hnc.hnc import Integral_Equation_Solver as IET
from hnc.hnc.constants import *
from .grids import NonUniformGrid
from .solvers import jacobi_relaxation, sor, gmres_ilu, tridiagsolve, Ndiagsolve
from .physics import FermiDirac, ThomasFermi, χ_Lindhard, χ_TF, More_TF_Zbar, Fermi_Energy, n_from_rs, Debye_length

class Plasma():
    """
    Basics of an Atom 
    """
    def __init__(self, Z, A, Ti, Te, rs, name=''):
        self.A = A
        self.Z = Z
        self.Ti = Ti
        self.Te = Te
        self.rs = rs
        self.name = name

class AverageAtom(Plasma):
    def __init__(self, Z, A, Ti, Te, rs, R, **kwargs):
        super().__init__(Z, A)
        self.R = R
        self.name = kwargs.get('name', '')
        self.initialize_grid(kwargs)
        self.set_physical_params()
        # Additional initializations
        self.initialize_density_and_potentials(kwargs)

    def initialize_grid(self, kwargs):
        self.N_stencil_oneside = kwargs.get('N_stencil_oneside', 2)
        self.grid = NonUniformGrid(2e-2, self.R, kwargs.get('Npoints', 100), self.rs, spacing='quadratic', N_stencil_oneside=self.N_stencil_oneside)
        self.rws_index = np.argmin(np.abs(self.grid.xs - self.rs))

    def set_physical_params(self):
        self.ni_bar = 1 / (4/3 * np.pi * self.rs**3)  # Average Ion Density of plasma
        self.ne_bar = self.Z * self.ni_bar  # Average Electron Density of plasma
        self.EF = Fermi_Energy(self.ne_bar)
        self.kF = (2 * self.EF)**(1/2)
        self.λTF = Debye_length(self.Te, self.ni_bar, self.Z)
        self.kTF = 1 / self.λTF
        self.κ = self.λTF * self.rs
        self.φ_κ_screen = self.κ
        self.Γ = self.Z**2 / (self.rs * self.Ti)
        self.make_χee()

    def make_χee(self):
        self.χee = lambda k: χ_Lindhard(k, self.kF) if self.χ_type == 'Lindhard' else χ_TF(k, self.kF)
        self.χee = np.vectorize(self.χee)

    def solve(self, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def initialize_density_and_potentials(self, kwargs):
        self.φe = self.Z / self.grid.xs * np.exp(-self.kTF * self.grid.xs) - (self.Z / self.grid.xs - self.Z / self.grid.xmax)
        self.ne = self.ne_bar * np.ones_like(self.grid.xs)
        self.μ_init = kwargs.get('μ_init', None)
        self.fixed_Zstar = kwargs.get('fixed_Zstar', False)
        self.use_full_ne_for_nf = kwargs.get('use_full_ne_for_nf', False)
        self.χ_type = kwargs.get('χ_type', 'Lindhard')
        self.set_μ_initial()

    def set_μ_initial(self):
        if self.μ_init is None:
            self.set_μ_neutral()
        else:
            self.μ = self.μ_init

    def set_μ_neutral(self):
        self.μ = self.get_μ_neutral()

    def get_μ_neutral(self):
        min_μ = lambda μ: abs(self.Z + self.grid.integrate_f(self.ρi - self.get_ne_TF(self.φe, self.ne, μ, self.ne_bar)))**2
        μ_guess = self.μ if self.μ_init is None else self.μ_init
        root_and_info = root(min_μ, μ_guess, tol=1e-12)
        μ = root_and_info['x'][0]
        return μ

    def get_ne_TF(self, φe, ne, μ, ne_bar):
        eta = self.get_eta_from_sum(φe, ne, μ, ne_bar)
        ne = self.fast_n_TF(eta)
        return ne

    def get_eta_from_sum(self, φe, ne, μ, ne_bar):
        βVeff = self.get_βVeff(φe, ne, ne_bar)
        eta = μ / self.Te - βVeff
        return eta

    def get_βVeff(self, φe, ne, ne_bar):
        βVeff = (-φe - self.φion + self.vxc_f(ne) - self.vxc_f(ne_bar)) / self.Te if not self.ignore_vxc else (-φe - self.φion) / self.Te
        return βVeff

    def make_ρi(self):
        self.ni = self.ni_bar * self.gii
        self.ρi = self.ni * self.Z
        self.Qion = self.Z + self.grid.integrate_f(self.ρi)

    def solve_iet(self, **kwargs):
        self.iet = IET(1, self.Γ, 3/(4*np.pi), self.Ti, 1, kappa=self.κ, R_max=self.iet_R_over_rs, N_bins=self.iet_N_bins, dst_type=3)
        self.iet.HNC_solve(**kwargs)
        self.gii_from_iet = interp1d(self.iet.r_array*self.rs, self.iet.h_r_matrix[0,0] + 1, bounds_error=False, fill_value='extrapolate', kind='cubic')(self.grid.xs)
    
    def make_gii(self, **kwargs):
        if self.rs == self.R:
            self.gii = np.zeros_like(self.grid.xs)
        elif self.gii_init_type == 'iet':
            self.solve_iet(**kwargs)
            gii_iet = self.iet.h_r_matrix[0,0] + 1
            self.gii = interp1d(self.iet.r_array*self.rs, gii_iet, bounds_error=False, fill_value='extrapolate')(self.grid.xs)
        else:
            self.gii = np.ones_like(self.grid.xs) * np.heaviside(self.grid.xs - self.rs, 1)
    
    def initialize_ne(self):
        self.ne = self.ne_bar * np.ones_like(self.grid.xs)
        self.set_μ_neutral()
        eta_approx = self.μ / self.Te - self.get_βVeff(self.φe, self.ne, self.ne_bar)
        self.ne_init_core = self.fast_n_TF(eta_approx)
        if self.rs == self.R:
            self.ne = self.ne_init_core
        else:
            transition_func = np.exp(-0.5 * (self.grid.xs / self.rs)**2)
            self.ne_init_outer = (1 - transition_func) * self.ρi
            netQ_outer = self.grid.integrate_f(self.ρi - self.ne_init_outer)
            remaining_Q = self.Z + netQ_outer
            self.ne_init_core *= transition_func
            self.ne_init_core *= remaining_Q / self.grid.integrate_f(self.ne_init_core)
            self.ne = self.ne_init_core + self.ne_init_outer
            self.set_μ_neutral()
        self.n_b, self.n_f = self.grid.zeros.copy(), self.grid.zeros.copy()

    def solve_TF(self, verbose=False, **kwargs):
        if verbose:
            print("Beginning self-consistent electron solver.")
            print("_________________________________")
        self.initialize_ne()
        self.φe, poisson_err = self.get_φe_screened(self.ρi - self.ne)
        poisson_err = np.mean(poisson_err)
        converged = False
        n = 0
        while not converged and n < int(kwargs.get('nmax', 1e4)):
            Q_old = self.get_Q()
            self.φe, poisson_err = self.get_φe_screened(self.ρi - self.ne)
            self.update_ne_picard(kwargs.get('picard_alpha', 1e-2))
            if not self.fixed_Zstar and n > kwargs.get('n_wait_update_Zstar', 100):
                self.update_newton_Zstar()
            if n % 10 == 0 or n < 5:
                self.set_μ_neutral()
            else:
                self.update_μ_newton(alpha1=1e-3)
            rho_err = self.rel_error(self.get_ne_TF(self.φe, self.ne, self.μ, self.ne_bar), self.ne, weight=4*np.pi*self.grid.xs**2, abs=True)
            Q = self.get_Q()
            delta_Q = Q - Q_old
            new = self.μ, np.mean(self.ne), np.mean(self.φe)
            change = self.L2_change(new, old)
            if change < kwargs.get('tol', 1e-4) and abs(rho_err) < kwargs.get('tol', 1e-4):
                converged = True
            n += 1

        print("TF Iteration {0}".format(n))
        print("	μ = {0:10.9e}, change: {1:10.9e}".format(self.μ, np.abs(1 - self.μ / old[0])))
        print("	φe Err = {0:10.3e}, φe change = {1:10.3e}".format(poisson_err, self.rel_error(self.φe_list[-1], self.φe_list[-2])))
        print("	ne Err = {0:10.3e}, ne change = {1:10.3e}".format(rho_err, self.rel_error(self.ne_list[-1], self.ne_list[-2])))
        print("	Q = {0:10.3e} -> {1:10.3e}".format(Q_old, Q))
        print("	Zstar guess = {0:10.3e}. Current Zstar: {1:10.3e}".format(self.new_Zstar_guess, self.Zstar))
        print("	Change = {0:10.3e}".format(change))
        self.poisson_err = poisson_err
        self.rho_err = rho_err
        self.Q = Q
        return converged


