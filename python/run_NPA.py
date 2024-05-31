import numpy as np
from atomic_forces.average_atom.python.average_atom_geometric import NeutralPseudoAtom as NPA

eV = 0.0367512 # So 4 eV = 4 * 0.036.. in natural units
Kelvin = 8.61732814974493e-5*eV #Similarly, 1 Kelvin = 3.16... in natural units 
π = np.pi

# ALL units in Hartree A.U.
Z = 13
A = 28
Temps= np.array([1,5,10,50,100,500,1000])*eV     # Temperature of plasma
rs = 5.441299912691573   # Wigner-Seitz of nucleus, or ion sphere radius
R = 10 * rs  # Correlation Radius

for T in Temps:
	Al = NPA(Z, A, T, rs, R, initialize=True, TFW=False, Zstar_init = 3, Npoints=200, name='Al',ignore_vxc=False)
	Al.print_metric_units()
	Al.solve_NPA(verbose=False, tol=1e-5)
	Al.save_data()
	# Al.make_plots(show=True)