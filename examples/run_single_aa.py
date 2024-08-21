import numpy as np
import pandas as pd

from average_atom.core.average_atom_kspace import NeutralPseudoAtom as NPA
from average_atom.core.misc import jacobi_relaxation, sor
from average_atom.core.grids import FourierGrid
from average_atom.core.physics import ThomasFermi, FermiDirac

from hnc.hnc.constants import *
from hnc.hnc.misc import rs_from_n, n_from_rs

from time import time

# Aluminuma
name='Al'
Z, A = 13, 27 

ρ = 2.699 #g/cc, the solid density of aluminum 
ni_cc = ρ/(A*amu_to_AU*AU_to_g)
ni_AU = ni_cc*invcc_to_AU
rs = rs_from_n(ni_AU)

Ti_eV = 1 
Te_eV = 1

Te_AU = Te_eV*eV_to_AU
Ti_AU = Ti_eV*eV_to_AU


####
R = 10*rs
ignore_vxc = True
fixed_Zstar = False
Zstar_init = 3.5 # 'More'

# Setup AA paramaters
aa_kwargs = {'initialize':True, 'gradient_correction':None, 'μ_init' : 0.158, 'Zstar_init' : Zstar_init, 'Npoints':1000,
             'name':name, 'ignore_vxc':ignore_vxc, 'fixed_Zstar':fixed_Zstar, 'iet_R_over_rs':R/rs, 'iet_N_bins':5000, 'use_full_ne_for_nf':False,
            'gii_init_type': 'step'}

# Setup NPA parameters
npa_kwargs = aa_kwargs.copy()
npa_kwargs.update({'Npoints':10000, 'iet_N_bins':10000, 'gii_init_type': 'iet', 'grid_spacing':'linear'})


# Start Evaluation
t00 = time()

t0 = time()
aa = NPA(Z, A, Ti_AU, Te_AU, rs, rs, **aa_kwargs)
print(f"Time to setup AA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")
t0 = time()
npa = NPA(Z, A, Ti_AU, Te_AU, rs, R, **npa_kwargs)                           
print(f"Time to setup NPA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")

t0 = time()
aa.solve_TF(verbose=False, picard_alpha=0.2, nmax = 5, tol=1e-9)
aa.set_uii_eff()
print(f"Time to solve AA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")

t0 = time()
npa.solve_TF(verbose=False, picard_alpha=1e-2, tol=1e-9, nmax = 5, n_wait_update_Zstar= 500)
npa.set_uii_eff()
print(f"Time to solve NPA: {time()-t0:0.3e} [s], time so far: {time()-t00:0.3e} [s]")

aa.save_data()
npa.save_data()
print("Saved.")