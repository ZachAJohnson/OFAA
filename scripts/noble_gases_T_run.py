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
      def __init__(self, name, Z, A, T_K, Pbar_initial):
            self.name = name
            self.Z = Z
            self.A = A
            self.T_K = T_K
            self.Te = T_K*K_to_AU
            self.Ti = T_K*K_to_AU
            
            self.Pbar_initial = Pbar_initial
            self.ni_cc = Pbar_initial*bar_to_AU/(T_room_K*K_to_AU)*AU_to_invcc
            self.ni_AU = self.ni_cc*invcc_to_AU
            self.rs = rs_from_n(self.ni_AU)


def run_case(case_info):
      run_ZJ_IS = True

      case, case_number = case_info
      # Redirect stdout to a file with a unique name
      if not os.path.exists(savefolder):
            os.makedirs(savefolder)

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
            print("T[K]: ", Te_AU*AU_to_K)
            aa_ZJ  = AverageAtomFactory.create_model("ZJ_ISModel", Z, A, Ti_AU, Te_AU, rs, R, 
                  name=name, ignore_vxc=ignore_vxc, xc_type=xc_type, Npoints=2000, rmin=1e-4, N_stencil_oneside = 2, savefolder=savefolder)
            print(f"Time to setup correlation sphere (IS) model: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
            
            t0 = time()
            aa_ZJ.solve(picard_alpha=0.2, verbose=True)
            aa_ZJ.set_uii_eff()
            print(f"Time to solve CS model: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
            aa_ZJ.save_data()
            print("CS model saved.")
            aa_ZJ.print_EOS()

      aa_ZJ.get_E()
      aa_ZJ.get_P()
      sys.stdout.close()
      return case.ni_AU, case.T_K, aa_ZJ.Ke, aa_ZJ.Ue, aa_ZJ.Ee, aa_ZJ.Pe, aa_ZJ.Zstar

name='Ar'
T_room_K = 290
P_bar = 25
TK_peak = 17.761029411764707e3 # 0.008097165991902834 ns?
Z, A = 18, 39.948

T_K_list = np.geomspace(T_room_K, 2*TK_peak, num=100)# Kelvin

# Setup parameters
savefolder = os.path.join(PACKAGE_DIR,"data",f"{name}_{datetime.today().strftime('%Y-%m-%d')}")
case_list = [aa_case(name, Z, A, T_K, P_bar) for i, T_K in enumerate(T_K_list)]

#### Arguments same for all cases
ignore_vxc  = False
xc_type = 'simple'
fixed_Zstar = False

# Run all cases
pool_obj = multiprocessing.Pool(processes=8)
EOS_results = pool_obj.map(run_case, [(case, i) for i, case in enumerate(case_list)])
pool_obj.close()
pool_obj.join()

EOS_data = np.array(EOS_results)
header = f"   {'ni[AU]':15} {'T[K]':15} {'Ke[AU]':15} {'Ue[AU]':15} {'Ee[AU]':15} {'Pe[AU]':15} {'Zbar':15}"
np.savetxt(os.path.join(savefolder, f'{name}_EOS.dat' ), EOS_data, delimiter = ' ', header=header, fmt='%15.6e', comments='')

