import numpy as np
import pandas as pd

from average_atom.core.average_atom_kspace import NeutralPseudoAtom as NPA
from average_atom.core.misc import jacobi_relaxation, sor
from average_atom.core.grids import FourierGrid
from average_atom.core.physics import ThomasFermi, FermiDirac

from hnc.hnc.constants import *
from hnc.hnc.misc import rs_from_n, n_from_rs

from time import time


class aa_case:
      def __init__(self, name, Z, A, T_AU, ρ_gpercc):
            self.name = name
            self.Z = Z
            self.A = A
            self.Te = T_AU
            self.Ti = T_AU
            self.ρ_gpercc = ρ_gpercc
            self.ni_cc = ρ_gpercc/(A*amu_to_AU*AU_to_g)
            self.ni_AU = self.ni_cc*invcc_to_AU
            self.rs = rs_from_n(self.ni_AU)


T_list = np.array([0.5,2,5])*eV_to_AU
# Cases in "Efficacy of RPPP..." Stanek ~ 2020
C1, C2, C3 = [aa_case(f"C{i+1}", 6, 12.011, T, 2.267 ) for i, T in enumerate(T_list)]
Al1, Al2, Al3 = [aa_case(f"Al{i+1}", 13, 26.981539, T, 2.7 ) for i, T in enumerate(T_list)]
V1, V2, V3 = [aa_case(f"V{i+1}", 23, 50.9415, T, 6.11 ) for i, T in enumerate(T_list)]
Au1, Au2, Au3 = [aa_case(f"Au{i+1}", 79, 196.96657, T, 19.30 ) for i, T in enumerate(T_list)]

case_list = [C1, C2, C3,  Al1, Al2, Al3,  V1, V2, V3,  Au1, Au2, Au3]
case_list = [Au1, Au2, Au3]

#### Arguments same for all cases
ignore_vxc = True
fixed_Zstar = False

# Setup AA paramaters
aa_kwargs = {'initialize':True, 'gradient_correction':None, 'μ_init' : 0.158, 'Zstar_init' : 'More', 'Npoints':1000,
            'ignore_vxc':ignore_vxc, 'fixed_Zstar':fixed_Zstar, 'iet_N_bins':5000, 'use_full_ne_for_nf':False,
            'gii_init_type': 'step'}

# Setup NPA parameters
npa_kwargs = aa_kwargs.copy()
npa_kwargs.update({'Npoints':10000, 'iet_N_bins':10000, 'gii_init_type': 'iet', 'grid_spacing':'linear'})


# Loop over all cases
for case in case_list:

      print(f"n===========================================\n===========================================")
      print(f"Starting case: {case.name}\n")
      t00 = time()
      name, Z, A, Ti_AU, Te_AU, rs = case.name, case.Z, case.A, case.Ti, case.Te, case.rs
      R = 10*rs

      aa_kwargs.update( {'iet_R_over_rs':R/rs,'name':name} )
      npa_kwargs.update( {'iet_R_over_rs':R/rs,'name':name} )

      # Start Evaluation
      t0 = time()
      aa = NPA(Z, A, Ti_AU, Te_AU, rs, rs, **aa_kwargs)
      print(f"Time to setup AA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
      
      t0 = time()
      aa.solve_TF(verbose=False, picard_alpha=0.2, nmax = 5000, tol=1e-9, n_wait_update_Zstar= 100)
      aa.set_uii_eff()
      print(f"Time to solve AA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
      
      t0 = time()
      npa = NPA(Z, A, Ti_AU, Te_AU, rs, R, **npa_kwargs)                           
      print(f"Time to setup NPA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")


      # Better initial guess 
      transition_func = 1/(1 + np.exp(-(npa.grid.xs-2*npa.rs)/(2*(rs/4)**2)))
      aa_ne_extended = np.zeros_like(npa.ne)
      aa_ne_extended[:len(aa.ne)] = aa.ne.copy()
      aa_ne_extended[len(aa.ne):] = aa.ne.copy()[-1]
      npa.ne = aa_ne_extended * (1-transition_func) + transition_func * npa.ρi 
      npa.μ = aa.μ
      npa.Zstar = aa.Zstar

      t0 = time()
      npa.solve_TF(verbose=False, picard_alpha=1e-2, tol=1e-9, nmax = 10000, n_wait_update_Zstar= 500)
      npa.set_uii_eff()
      print(f"Time to solve NPA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")

      aa.save_data()
      npa.save_data()
      print("Saved.")