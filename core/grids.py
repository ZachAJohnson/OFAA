import numpy as np
from scipy import fftpack
from scipy.integrate import simpson

π = np.pi


def weights(z, x, n, m):
    # From Bengt Fornbergs (1998) SIAM Review paper.
    # Input Parameters:
    # z - location where approximations are to be accurate
    # x - grid point locations, found in x[0:n]
    # n - one less than total number of grid points; n must not exceed the parameter nd
    # m - highest derivative for which weights are sought
    # Output:
    # c - weights at grid locations x[0:n] for derivatives of order 0:m
    
    c = np.zeros((n+1, m+1))
    c1 = 1.0
    c4 = x[0] - z
    for k in range(m+1):
        for j in range(n+1):
            c[j, k] = 0.0
    
    c[0, 0] = 1.0
    for i in range(1, n+1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i-1, k-1] - c5 * c[i-1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i-1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k-1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


class OneDGrid():
	"""	
	Simple 1-D grid base class
	"""
	def __init__(self, xmin, xmax, Nx):
		self.xmin = xmin
		self.xmax = xmax
		self.Nx = Nx

		self.make_grid()

		self.zeros = np.zeros((self.Nx)) #Must be .copy() to work
		self.ones  = np.ones ((self.Nx)) #Must be .copy() to work

class FourierGrid():

	def __init__(self, xmin, xmax, Nx, dst_type=3):
		self.xmin = xmin
		self.xmax = xmax
		self.Nx = Nx

		self.dst_type = dst_type
		self.make_k_r_spaces()

		self.zeros = np.zeros((self.Nx)) #Must be .copy() to work
		self.ones  = np.ones ((self.Nx)) #Must be .copy() to work


	# R and K space grid and Fourier Transforms
	def make_k_r_spaces(self):

		if  self.dst_type==1: 
			self.xs = np.linspace(0, self.xmax, num=self.Nx+1)[1:]
			self.dx = self.xs[1]-self.xs[0]
			self.ks = np.array([π*(l+1)/self.xmax for l in range(self.Nx)] ) #Type 1
			self.dk = self.ks[1]-self.ks[0] 

		elif self.dst_type==2:
			self.xs = np.linspace(0, self.xmax, num=self.Nx+1)[1:]
			self.dx = self.xs[1]-self.xs[0]
			self.xs -= self.dx/2 #So now theoretically r_array[1/2] would be zero
			self.ks = np.array([π*(l+1)/self.xmax for l in range(self.Nx)] ) #Type 1
			self.dk = self.ks[1]-self.ks[0] 

		elif self.dst_type==3:   
			self.xs = np.linspace(0, self.xmax, num=self.Nx+1)[1:]
			self.dx = self.xs[1]-self.xs[0]
			self.ks = np.array([π*(l+0.5)/self.xmax for l in range(self.Nx)] ) #Type 3
			self.dk = self.ks[1]-self.ks[0] 

		elif self.dst_type==4:       
			self.xs = np.linspace(0, self.xmax, num=self.Nx+1)[1:]
			self.dx = self.xs[1]-self.xs[0]
			self.xs -= self.dx/2 #So now theoretically r_array[1/2] would be zero
			self.ks = np.array([π*(l+0.5)/self.xmax for l in range(self.Nx)] ) #Type 3
			self.dk = self.ks[1]-self.ks[0] 

        
		self.fact_r_2_k = 2 * np.pi * self.dx
		self.fact_k_2_r = self.dk / (4. * np.pi**2)

		self.vols = 4*π*self.xs**2*self.dx

	def FT_r_2_k(self, input_array):
	    from_dst = self.fact_r_2_k * fftpack.dst(self.xs * input_array, type=self.dst_type)
	    return from_dst / self.ks

	def FT_k_2_r(self, input_array):
	    from_idst = self.fact_k_2_r * fftpack.idst(self.ks * input_array, type=self.dst_type)
	    return from_idst / self.xs


	def integrate_f(self, f, begin_index = 0, end_index=None):
		integrated_slice = slice(begin_index, end_index)
		return np.sum( (f*self.vols)[integrated_slice])

	def dfdx(self, f):
		"""
		Fourth order derivative of some function. Fills end with zeros.
		df ~ O(dx^5) 
		"""
		ffull = self.zeros.copy()
		#middle
		mid_coeffs = [1/12., -2/3, 0, 2/3, -1/12]
		mid_shifts = [-2, -1, 0, 1, 2]
		ffull[2:-2] = (np.sum([c*np.roll(f,-shift) for c, shift in zip(mid_coeffs, mid_shifts)], axis=0)/self.dx)[2:-2]
		
		#EDges
		ffull[0]  = (-0.25*f[4] + 4/3*f[3] -3*f[2] +4*f[1] - 25/12*f[0])/self.dx
		ffull[1]  = (-0.25*f[5] + 4/3*f[4] -3*f[3] +4*f[2] - 25/12*f[1])/self.dx
		ffull[-1] = (0.25*f[-5] - 4/3*f[-4] +3*f[-3] -4*f[-2] + 25/12*f[-1])/self.dx
		ffull[-2] = (0.25*f[-6] - 4/3*f[-5] +3*f[-4] -4*f[-3] + 25/12*f[-2])/self.dx

		return ffull

	def d2fdx2(self, f):
		"""
		Second order derivative of some function f. Fills end with zeros
		d2f ~ O(dx^3)
		"""
		ffull = self.zeros.copy()
		ffull[1:-1] = (f[2:] + f[:-2] - 2*f[1:-1])/self.dx**2
		ffull[0] = (f[0] - 2*f[1] + f[2])/self.dx**2
		ffull[-1] = (f[-1] - 2*f[-2] + f[-3])/self.dx**2
		return ffull

class LinearGrid(OneDGrid):

	def __init__(self, xmin, xmax, Nx):
		super().__init__(xmin,xmax,Nx)

	def make_grid(self):
		self.xs = np.linspace(self.xmin, self.xmax, num=self.Nx+1, endpoint=True)[1:]
		self.dx = self.xs[1]-self.xs[0]
		self.bulk_indcs = slice(1,-1)
		self.grid_shape = self.Nx

	def integrate_f(self, f, end_index= None):
		integration_region = slice(None, end_index)
		return simps((f*self.vols)[integration_region], x = self.xs[integration_region])

	def dfdx(self, f):
		"""
		Fourth order derivative of some function. Fills end with zeros.
		df ~ O(dx^5) 
		"""
		ffull = self.zeros.copy()
		#middle
		mid_coeffs = [1/12., -2/3, 0, 2/3, -1/12]
		mid_shifts = [-2, -1, 0, 1, 2]
		ffull[2:-2] = (np.sum([c*np.roll(f,-shift) for c, shift in zip(mid_coeffs, mid_shifts)], axis=0)/self.dx)[2:-2]
		
		#EDges
		ffull[0]  = (-0.25*f[4] + 4/3*f[3] -3*f[2] +4*f[1] - 25/12*f[0])/self.dx
		ffull[1]  = (-0.25*f[5] + 4/3*f[4] -3*f[3] +4*f[2] - 25/12*f[1])/self.dx
		ffull[-1] = (0.25*f[-5] - 4/3*f[-4] +3*f[-3] -4*f[-2] + 25/12*f[-1])/self.dx
		ffull[-2] = (0.25*f[-6] - 4/3*f[-5] +3*f[-4] -4*f[-3] + 25/12*f[-2])/self.dx

		return ffull

	def d2fdx2(self, f):
		"""
		Second order derivative of some function f. Fills end with zeros
		d2f ~ O(dx^3)
		"""
		ffull = self.zeros.copy()
		ffull[1:-1] = (f[2:] + f[:-2] - 2*f[1:-1])/self.dx**2
		ffull[0] = (f[0] - 2*f[1] + f[2])/self.dx**2
		ffull[-1] = (f[-1] - 2*f[-2] + f[-3])/self.dx**2
		return ffull

class NonUniformGrid(OneDGrid):
	def __init__(self,   xmin, xmax, Nx, rs, spacing='quadratic', N_stencil_oneside=2):
		self.rs = rs
		self.spacing = spacing
		self.N_stencil_oneside = N_stencil_oneside
		super().__init__(xmin, xmax, Nx)
		self.make_dndx_matrices_byFornberg()

	

	def make_grid(self, frac_geometric = 0.25):
		if self.spacing=='quadratic':
			ε_sqrt = np.linspace(np.sqrt(self.xmin), np.sqrt(self.xmax), num=self.Nx, endpoint=True )
			self.xs = ε_sqrt**2
			dε = ε_sqrt[1]-ε_sqrt[0]
			cell_boundary_points = np.linspace(np.sqrt(self.xmin) - dε/2 , np.sqrt(self.xmax)+dε/2, num = self.Nx+1, endpoint=True )**2
			
		elif self.spacing=='geometric':
			ε_geom = np.linspace(np.log(self.xmin), np.log(self.xmax), num=self.Nx, endpoint=True)
			self.xs = np.exp(ε_geom)
			dε = ε_geom[1] - ε_geom[0]
			cell_boundary_points = np.exp(np.linspace(np.log(self.xmin)-dε/2, np.log(self.xmax)+dε/2, num=self.Nx+1, endpoint=True))
		elif self.spacing == 'linear':
			ε_lin = np.linspace(self.xmin, self.xmax, num=self.Nx, endpoint=True)
			self.xs = ε_lin
			dε = ε_lin[1] - ε_lin[0]
			cell_boundary_points = np.linspace(self.xmin - dε/2, self.xmax + dε/2, num=self.Nx+1, endpoint=True)

		
		self.cells = cell_boundary_points
		self.vols = 4/3*π*(self.cells[1:]**3 - self.cells[:-1]**3)
		self.dx = self.xs[1:]-self.xs[:-1]
		self.dL = self.cells[1:] - self.cells[:-1]
		self.bulk_indcs = slice(1,-1)
		self.grid_shape = self.Nx
	
	def integrate_f(self, f, end_index=None):
		# f_geomfactor = 4*π*f*self.xs**2
		# return np.sum(  (0.5*(f_geomfactor[1:] + f_geomfactor[:-1])*self.dx )[:end_index] )
		# integrated_slice = slice(None, end_index)
		if end_index==None:
			actual_end_index = end_index
		else:
			actual_end_index = end_index + 1
		if end_index == 0:
			return 0
		else:
			return simpson( (f*self.xs**2*4*π)[:actual_end_index], x=self.xs[:actual_end_index])


	def dfdx(self, f):
		"""
		First order derivative of some function.
		"""
		return self.A_dfdx.dot(f)

	def d2fdx2(self, f):
		"""
		Second order derivative of some function f.
		"""
		return self.A_d2fdx2.dot(f)

	def laplace(self, f):
		return self.A_laplace.dot(f)
	
	def make_dndx_matrices_byFornberg(self):
		num_left, num_right = self.N_stencil_oneside, self.N_stencil_oneside
		A_1 = np.zeros((self.Nx, self.Nx)) # Is N x N, dfdx
		A_2 = np.zeros((self.Nx, self.Nx)) # Is N x N, d2fdx2

		dx= self.dx
		x = self.xs
		N = len(self.dx)
		
		for i in np.arange(self.Nx)[num_left:-num_right]: # tri-diagonal index
			weight_slice = slice(i-num_left, i+num_right+1)
			center_weights = weights(x[i], x[weight_slice], num_left+num_right, 2).T
			A_1[i, weight_slice] = center_weights[1] # first derivative coefficients
			A_2[i, weight_slice] = center_weights[2] # second derivative coefficients
		
		for i in range(0, num_left):
			total_points = num_right + i + 1
			weight_slice = slice(None,total_points)
			left_edge_weights = weights(x[i], x[weight_slice], total_points-1, 2).T
			A_1[i, weight_slice ] = left_edge_weights[1] # first derivative coefficients 
			A_2[i, weight_slice ] = left_edge_weights[2] # second derivative coefficients

		for i in range(0, num_right):
			total_points = num_left + i + 1
			weight_slice = slice(-total_points, None)
			right_edge_weights = weights(x[N-i], x[weight_slice], total_points-1, 2).T
			A_1[N-i, weight_slice ] = right_edge_weights[1] # first derivative coefficients 
			A_2[N-i, weight_slice ] = right_edge_weights[2] # second derivative coefficients

		self.A_dfdx = A_1
		self.A_d2fdx2 = A_2
		self.A_laplace = 2/x[:,np.newaxis] * A_1 + A_2


	# def matrix_d2fdx2(self):
	# 	A = np.zeros((self.Nx, self.Nx)) # Is N x N
	# 	Abulk = A.copy()

	# 	dx= self.dx
	# 	x = self.xs

	# 	i = np.arange(self.Nx)[1:-1] # tri-diagonal index

	# 	A[i, i+1] =   2/dx[i]  /(dx[i] + dx[i-1])   #1 above diagonal
	# 	A[i, i]   =  -2/(dx[i]*dx[i-1])           #diagonal
	# 	A[i, i-1] =   2/dx[i-1]/(dx[i] + dx[i-1])   #1 below diagonal
	
	# 	Abulk[1:-1,:] = A[1:-1,:]
	# 	return Abulk

	# def matrix_dfdx(self):
	# 	A = np.zeros((self.Nx, self.Nx)) # Is N x N
	# 	Abulk = A.copy()

	# 	dx= self.dx
	# 	x = self.xs

	# 	i = np.arange(self.Nx)[1:-1] # tri-diagonal index

	# 	A[i, i+1] =  dx[i-1]/dx[i]/(dx[i] + dx[i-1]) #1 above diagonal
	# 	A[i, i]   = (dx[i]-dx[i-1])/(dx[i]*dx[i-1])  #diagonal
	# 	A[i, i-1] = -dx[i]/dx[i-1]/(dx[i] + dx[i-1]) #1 below diagonal
	
	# 	Abulk[1:-1,:] = A[1:-1,:]
	# 	return Abulk


	# def matrix_d2fdx2(self):
	# 	A = np.zeros((self.Nx, self.Nx)) # Is N x N
	# 	Abulk = A.copy()

	# 	dx = self.dx
	# 	x = self.xs

	# 	i = np.arange(self.Nx)[2:-2] # Five-diagonal index

	# 	A[i, i+2] =   1 / (dx[i] * (dx[i] + dx[i-1] + dx[i-2]))  # 2 above diagonal
	# 	A[i, i+1] =  -4 / (dx[i] * (dx[i] + dx[i-1]))           # 1 above diagonal
	# 	A[i, i]   =   6 / (dx[i] * dx[i-1])                     # diagonal
	# 	A[i, i-1] =  -4 / (dx[i-1] * (dx[i] + dx[i-1]))         # 1 below diagonal
	# 	A[i, i-2] =   1 / (dx[i-1] * (dx[i] + dx[i-1] + dx[i-2])) # 2 below diagonal

	# 	Abulk[2:-2, :] = A[2:-2, :]
	# 	return Abulk

	# def matrix_dfdx(self):
	# 	A = np.zeros((self.Nx, self.Nx))  # Is N x N
	# 	Abulk = A.copy()

	# 	dx = self.dx
	# 	x = self.xs

	# 	i = np.arange(self.Nx)[2:-2]  # Five-diagonal index

	# 	for j in i:
	# 		dx_m2 = x[j-2] - x[j]
	# 		dx_m1 = x[j-1] - x[j]
	# 		dx_0 = x[j] - x[j]
	# 		dx_1 = x[j+1] - x[j]
	# 		dx_2 = x[j+2] - x[j]

	# 		A[j, j-2] =  1 / (2 * dx_m2 * (dx_m2 - dx_m1) * (dx_m2 - dx_0) * (dx_m2 - dx_1) * (dx_m2 - dx_2))
	# 		A[j, j-1] = -8 / (2 * dx_m1 * (dx_m1 - dx_m2) * (dx_m1 - dx_0) * (dx_m1 - dx_1) * (dx_m1 - dx_2))
	# 		A[j, j]   =  0  # This is typically zero for central differences.
	# 		A[j, j+1] =  8 / (2 * dx_1 * (dx_1 - dx_m2) * (dx_1 - dx_m1) * (dx_1 - dx_0) * (dx_1 - dx_2))
	# 		A[j, j+2] = -1 / (2 * dx_2 * (dx_2 - dx_m2) * (dx_2 - dx_m1) * (dx_2 - dx_0) * (dx_2 - dx_1))

	# 	Abulk[2:-2, :] = A[2:-2, :]
	# 	return Abulk

	# def matrix_laplacian(self):
	# 	"""
	# 	nalba^2 in spherical: 1/r^2 d/dr ( r^2 d/dr )
	# 	"""
	# 	dx= self.dx
	# 	x = self.xs
	# 	i = np.arange(self.Nx)[1:-1] # tri-diagonal index

	# 	# 2/r φ'
	# 	A1 = np.zeros((self.Nx, self.Nx)) # Is N x N
	# 	A1[i, i+1] = 2/x[i] *  dx[i-1]/dx[i]/(dx[i] + dx[i-1]) #1 above diagonal
	# 	A1[i, i]   = 2/x[i] * (dx[i]-dx[i-1])/(dx[i]*dx[i-1])  #diagonal
	# 	A1[i, i-1] = 2/x[i] * -dx[i]/dx[i-1]/(dx[i] + dx[i-1]) #1 below diagonal
		
	# 	# φ"		
	# 	A2 = self.matrix_d2fdx2()

	# 	A = A1 + A2 
	# 	Abulk =  np.zeros((self.Nx, self.Nx)) # Is N x N
	# 	Abulk[1:-1,:] = A[1:-1,:]
	# 	return Abulk

	# def matrix_laplacian_old(self):
	# 	"""
	# 	nalba^2 in spherical: 1/r^2 d/dr ( r^2 d/dr )
	# 	"""
	# 	dx= self.dx
	# 	x = self.xs
	# 	i = np.arange(self.Nx)[1:-1] # tri-diagonal index

	# 	# 2/r φ'
	# 	A1 = 2/x[:, np.newaxis]*self.matrix_dfdx()
	
	# 	# φ"		
	# 	A2 = self.matrix_d2fdx2()

	# 	A = A1 + A2 
	# 	Abulk =  np.zeros((self.Nx, self.Nx)) # Is N x N
	# 	Abulk[1:-1,:] = A[1:-1,:]
	# 	return Abulk

	# def matrix_laplacian(self):
	# 	"""
	# 	nalba^2 in spherical: 1/r^2 d/dr ( r^2 d/dr )
	# 	"""
	# 	dx= self.dx
	# 	x = self.xs
	# 	i = np.arange(self.Nx)[1:-1] # tri-diagonal index

	# 	#  r^2 d/dr
	# 	A1 = self.xs**2*self.matrix_dfdx()
		
	# 	# 1/r^2 d/dr		
	# 	A2 = self.xs**-2*self.matrix_dfdx()

	# 	A = np.dot(A2, A1)
	# 	Abulk =  np.zeros((self.Nx, self.Nx)) # Is N x N
	# 	Abulk[1:-1,:] = A[1:-1,:]


	# 	return Abulk