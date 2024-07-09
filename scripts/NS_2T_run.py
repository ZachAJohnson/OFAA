import numpy as np
import pandas as pd

from average_atom.core.average_atom import NeutralPseudoAtom as NPA
from average_atom.core.misc import jacobi_relaxation, sor
from average_atom.core.grids import NonUniformGrid
from average_atom.core.physics import ThomasFermi, FermiDirac

from hnc.hnc.constants import *
from hnc.hnc.misc import rs_from_n, n_from_rs

from time import time
import multiprocessing
import sys
import os

class aa_case:
      def __init__(self, name, Z, A, Ti_AU, Te_AU, ρ_gpercc):
            self.name = name
            self.Z = Z
            self.A = A
            self.Te = Te_AU
            self.Ti = Ti_AU
            self.ρ_gpercc = ρ_gpercc
            self.ni_cc = ρ_gpercc/(A*amu_to_AU*AU_to_g)
            self.ni_AU = self.ni_cc*invcc_to_AU
            self.rs = rs_from_n(self.ni_AU)


def run_case(case_info):
      run_aa = False
      run_npa = True
      

      case, case_number = case_info
      # Redirect stdout to a file with a unique name
      log_file = open(f'process_{case_number}.out', 'w', buffering=1)
      sys.stdout = log_file
      sys.stderr = log_file

      print(f"\n===========================================\n===========================================")
      t00 = time()
      name, Z, A, Ti_AU, Te_AU, rs = case.name, case.Z, case.A, case.Ti, case.Te, case.rs
      R = 10*rs
      print(f"Starting case: {case.name}: Te = {Te_AU*AU_to_eV:0.3f} eV, Ti = {Ti_AU*AU_to_eV:0.3f} eV \n")


      if run_aa:
            aa_kwargs = {'initialize':True, 'gradient_correction':None,'μ_init' : 0.158, 'Zstar_init' : 'More', 'rmin':1e-3 ,'Npoints':300, 
                    'name':name,'ignore_vxc':ignore_vxc, 'fixed_Zstar':fixed_Zstar, 'iet_R_over_rs':10, 'iet_N_bins':10000, 'use_full_ne_for_nf':False,
                   'grid_spacing':'geometric','N_stencil_oneside':2, 'χ_type':'TF'}

            aa_kwargs.update( {'iet_R_over_rs':R/rs,'name':name} )
            t0 = time()
            aa = NPA(Z, A, Ti_AU, Te_AU, rs, rs, **aa_kwargs)
            print(f"Time to setup AA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
            t0 = time()
            aa.solve_TF(verbose=True, picard_alpha=0.5, nmax = 2000, tol=1e-8, n_wait_update_Zstar= 50)
            aa.set_uii_eff()
            print(f"Time to solve AA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
            aa.save_data()
            print("AA saved.")

      if run_npa:
            npa_kwargs = {'initialize':True, 'gradient_correction':None,'μ_init' : 0.158, 'Zstar_init' : 'More', 'rmin':1e-3 ,'Npoints':1000, 
                          'name':name,'ignore_vxc':ignore_vxc, 'fixed_Zstar':fixed_Zstar, 'iet_R_over_rs':R/rs, 'iet_N_bins':10000, 'use_full_ne_for_nf':False,
                         'gii_init_type': 'step', 'grid_spacing':'quadratic','N_stencil_oneside':2, 'χ_type':'TF'}

            t0 = time()
            npa = NPA(Z, A, Ti_AU, Te_AU, rs, R, **npa_kwargs)                           
            print(f"Time to setup NPA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")

            if run_aa:
                  pass
                  # # Better initial guess 
                  # transition_func = 1/(1 + np.exp(-(npa.grid.xs-2*npa.rs)/(2*(rs/4)**2)))
                  # aa_ne_extended = np.zeros_like(npa.ne)
                  # aa_ne_extended[:len(aa.ne)] = aa.ne.copy()
                  # aa_ne_extended[len(aa.ne):] = aa.ne.copy()[-1]
                  # npa.ne = aa_ne_extended * (1-transition_func) + transition_func * npa.ρi 
                  # npa.μ = aa.μ
                  # npa.Zstar = aa.Zstar

            t0 = time()
            npa.solve_TF(verbose=True, picard_alpha=0.5, tol=1e-8, nmax = 2000, n_wait_update_Zstar= 50)
            npa.set_uii_eff()
            print(f"Time to solve NPA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
       
            npa.save_data()
            print("NPA saved.")

      sys.stdout.close()


# Setup parameters
T_ie_list = np.array([ 
      [1,1],
      [1,3],
      [1,10],
      [1,30],
      [3,30],
      [10,30],
      [30,30],
      ])*eV_to_AU

name='Al'
Z, A = 13, 26.981539
ρ_gpercc = 2.7
case_list = [aa_case(f"Al", Z, A, T_ie[0], T_ie[1], 2.7 ) for i, T_ie in enumerate(T_ie_list)]

#### Arguments same for all cases
ignore_vxc  = False
fixed_Zstar = False

# Run all cases
pool_obj = multiprocessing.Pool()#processes=8)
ans = pool_obj.map(run_case, [(case, i) for i, case in enumerate(case_list)])
pool_obj.close()
pool_obj.join()
