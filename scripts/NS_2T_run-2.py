import numpy as np
import pandas as pd

from average_atom.core.aa_types import AverageAtomFactory
from average_atom.core.misc import jacobi_relaxation, sor
from average_atom.core.grids import NonUniformGrid
from average_atom.core.physics import ThomasFermi, FermiDirac

from hnc.hnc.constants import *
from hnc.hnc.misc import rs_from_n, n_from_rs

from time import time
from datetime import datetime

import multiprocessing
import sys
import os
from average_atom.core.config import CORE_DIR, PACKAGE_DIR


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
      run_ZJ_IS = False
      run_ZJ_CS = False
      run_ZJ_SS = True
      

      case, case_number = case_info
      # Redirect stdout to a file with a unique name
      log_file = open(os.path.join(savefolder,f'process_{case_number}.out'), 'w', buffering=1)
      sys.stdout = log_file
      sys.stderr = log_file

      print(f"\n===========================================\n===========================================")
      t00 = time()
      name, Z, A, Ti_AU, Te_AU, rs = case.name, case.Z, case.A, case.Ti, case.Te, case.rs
      R = 10*rs
      print(f"Starting case: {case.name}: Te = {Te_AU*AU_to_eV:0.3f} eV, Ti = {Ti_AU*AU_to_eV:0.3f} eV \n")


      if run_ZJ_IS:
            t0 = time()
            aa_ZJ  = AverageAtomFactory.create_model("ZJ_ISModel", Z, A, Ti_AU, Te_AU, rs, R, 
                  name=name, ignore_vxc=False, Npoints=2000, rmin=1e-4, N_stencil_oneside = 2, savefolder=savefolder)
            print(f"Time to setup correlation sphere (IS) model: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
            
            t0 = time()
            aa_ZJ.solve(picard_alpha=0.5,verbose=True)
            aa_ZJ.set_uii_eff()

            print(f"Time to solve CS model: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
       
            aa_ZJ.save_data()

            print("CS model saved.")

      if run_ZJ_CS:
            t0 = time()
            aa_ZJ_cs  = AverageAtomFactory.create_model("ZJ_CSModel", Z, A, Ti_AU, Te_AU, rs, R, 
                  name=name, ignore_vxc=False, Npoints=2000, rmin=1e-4, N_stencil_oneside = 2, savefolder=savefolder)
            print(f"Time to setup correlation sphere (CS) model: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
            
            t0 = time()
            aa_ZJ_cs.solve(picard_alpha=0.5,verbose=True)
            aa_ZJ_cs.set_uii_eff()

            print(f"Time to solve CS model: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
       
            aa_ZJ_cs.save_data()
            print("CS model saved.")
      if run_ZJ_SS:
            t0 = time()
            aa_ZJ_SS  =  AverageAtomFactory.create_model("TFStarret2014", Z, A, Ti_AU, Te_AU, rs, R, 
                  name=name, ignore_vxc=False, Npoints=2000, rmin=1e-4, N_stencil_oneside = 2, savefolder=savefolder, χ_type='TF')
            print(f"Time to setup correlation sphere (CS) model: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
            
            aa_ZJ_SS.savefolder = os.path.join(savefolder, f"chi_{aa_ZJ_SS.χ_type}")

            t0 = time()
            aa_ZJ_SS.solve(picard_alpha=0.5,verbose=True)
            aa_ZJ_SS.set_uii_eff()

            print(f"Time to solve CS model: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
       
            aa_ZJ_SS.save_data()
            print("CS model saved.")

            print("Changing Ion parameters")
            aa_ZJ_SS.χ_type='Lindhard'
            aa_ZJ_SS.make_χee()
            aa_ZJ_SS.savefolder = os.path.join(savefolder, f"chi_{aa_ZJ_SS.χ_type}")
            aa_ZJ_SS.set_uii_eff()
            aa_ZJ_SS.save_data()


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


savefolder = os.path.join(PACKAGE_DIR,"data",f"Al_2T_{datetime.today().strftime('%Y-%m-%d')}")
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
