import numpy as np
# from atomic_forces.average_atom.python.average_atom import NeutralPseudoAtom as NPA
from atomic_forces.average_atom.python.average_atom_geometric import NeutralPseudoAtom as NPA
from atomic_forces.average_atom.python.average_atom_geometric import load_NPA

import sys
from time import time

eV = 0.0367512 # So 4 eV = 4 * 0.036.. in natural units
Kelvin = 8.61732814974493e-5*eV #Similarly, 1 Kelvin = 3.16... in natural units 
π = np.pi

# ALL units in Hartree A.U.
Z = 18#13
A = 49#28
T = float(input("Enter Temperature in eV: "))*eV     # Temperature of plasma
rs = 3#5.441299912691573   # Wigner-Seitz of nucleus, or ion sphere radius
R = 10 * rs  # Correlation Radius

t0 = time()

# # Course grid
Al = NPA(Z, A, T, rs, R, initialize=True, TFW=False, Zstar_init = 3 , Npoints=50, name='Al',ignore_vxc=False)
Al.print_metric_units()
# Al.solve_NPA( verbose=True, tol=1e-5)
# Al.save_data()
# print("Course grid Zstar={0:.3e}".format(Al.Zstar))
# Al.make_plots(show=False)

# Fine grid
# Al = load_NPA(Al.savefile, TFW=False, ignore_vxc=False)
# Al = load_NPA("/home/zach/plasma/atomic_forces/average_atom/data/Al_NPA_TFD_R5.4e+01_rs5.4e+00_T3.7e-01eV_Zstar0.0.dat", TFW=False, ignore_vxc=False)
# Al.new_grid(100)
# Al.solve_NPA(verbose=True, tol=1e-5)
Al.solve_HNC_NPA(verbose=False)
tf = time()
print("Total time: {0:.3e} s".format(tf-t0))

Al.save_data()
# print("Fine grid Zstar={0:.3e}".format(Al.Zstar))
Al.make_plots(show=True)
Al.hnc.plot_hnc()
Al.hnc.plot_potential()

