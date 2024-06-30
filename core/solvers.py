import numpy as np
import pandas as pd
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.linalg import solve_banded

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu

def tridiagsolve(A, b):
	N = np.shape(b)[0]
	Ab = np.zeros((3, N))
	
	Ab[0,1:] = np.diag(A,k=1)
	Ab[1]    = np.diag(A,k=0)
	Ab[2,:-1]= np.diag(A,k=-1)
	x = solve_banded((1,1), Ab, b)
	return x	

def Ndiagsolve(A, b, N_bands_above):
	N_bands = int(2*N_bands_above+1)
	N = np.shape(b)[0]
	Ab = np.zeros((N_bands, N))

	Ab[N_bands_above, :] = np.diag(A,k=0) # middle

	for i in range(1, N_bands_above + 1 ): # above
		Ab[N_bands_above - i, i:] = np.diag(A,k=i)

	for i in range(1, N_bands_above + 1 ): # below
		Ab[N_bands_above + i, :-i] = np.diag(A,k=-i)

	x = solve_banded((N_bands_above,N_bands_above), Ab, b)
	return x  

def gmres_ilu(A, b, tol=1e-4):
	# Convert the matrix A to Compressed Sparse Column (CSC) format
	A_csc = csc_matrix(A)

	# Compute the ILU decomposition of A
	ilu = spilu(A_csc)

	# Define the preconditioner as a callable function
	M = lambda x: ilu.solve(x)
	M = LinearOperator(A_csc.shape, ilu.solve)


	# Solve the linear system Ax = b using GMRES with the ILU preconditioner
	gmreskwargs = {'maxiter':int(1e4),'restart':int(20)}
	x, info = gmres(A_csc, b, M=M, tol=tol, **gmreskwargs)

	if info != 0:
		raise RuntimeError(f"GMRES failed to converge (info={info})")
	
	return x


# from atomic_forces.average_atom.python.average_atom_geometric import NeutralPseudoAtom  


def jacobi_relaxation(A, b, x0, tol=1e-6, nmax=100):
	n = len(A)
	x = x0.copy()
	x_prev = x0.copy()
	error = np.inf

	m=0
	while error > tol and nmax > m:
		for i in range(n):
			sum1 = np.dot(A[i, :i], x_prev[:i])
			sum2 = np.dot(A[i, i+1:], x_prev[i+1:])
			x[i] = (b[i] - sum1 - sum2) / A[i, i]

		error = np.linalg.norm(x - x_prev)
		x_prev = x.copy()
		# print(error)
		m+=1
	print(m, error)
	return x

def sor(A, b, x0, omega=1.1, tol=1e-6, nmax=100):
	n = len(A)
	x = x0.copy()
	x_prev = x0.copy()
	error = np.inf

	m=0
	while error > tol and nmax > m:
		for i in range(n):
			sum1 = np.dot(A[i, :i], x[:i])
			sum2 = np.dot(A[i, i+1:], x_prev[i+1:])
			x[i] = (1 - omega) * x_prev[i] + omega * (b[i] - sum1 - sum2) / A[i, i]

		error = np.linalg.norm(x - x_prev)
		x_prev = x.copy()
		# print(error)
		m+=1
	return x