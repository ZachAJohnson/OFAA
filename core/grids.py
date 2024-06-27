import numpy as np
from scipy import fftpack


π = np.pi


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
	def __init__(self,   xmin, xmax, Nx, rs, spacing='quadratic'):
		self.rs = rs
		self.spacing = spacing
		super().__init__(xmin, xmax, Nx)
	

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
	
	def integrate_f(self, f, begin_index = 0, end_index=None):
		# f_geomfactor = 4*π*f*self.xs**2
		# return np.sum(  (0.5*(f_geomfactor[1:] + f_geomfactor[:-1])*self.dx )[:end_index] )
		integrated_slice = slice(begin_index, end_index)
		return np.sum( (f*self.vols)[integrated_slice])


	def dfdx(self, f):
		"""
		First order derivative of some function.
		"""
		fgrad = np.gradient(f, self.xs, edge_order=1)
		
		return fgrad

	def d2fdx2(self, f):
		"""
		Second order derivative of some function f.
		"""
		fgrad2 = np.gradient(self.dfdx(f), self.xs, edge_order=2)
		return fgrad2

	def matrix_d2fdx2(self):
		A = np.zeros((self.Nx, self.Nx)) # Is N x N
		Abulk = A.copy()

		dx= self.dx
		x = self.xs

		i = np.arange(self.Nx)[1:-1] # tri-diagonal index

		A[i, i+1] =   2/dx[i]  /(dx[i] + dx[i-1])   #1 above diagonal
		A[i, i]   =  -2/(dx[i]*dx[i-1])           #diagonal
		A[i, i-1] =   2/dx[i-1]/(dx[i] + dx[i-1])   #1 below diagonal
	
		Abulk[1:-1,:] = A[1:-1,:]
		return Abulk

	def matrix_dfdx(self):
		A = np.zeros((self.Nx, self.Nx)) # Is N x N
		Abulk = A.copy()

		dx= self.dx
		x = self.xs

		i = np.arange(self.Nx)[1:-1] # tri-diagonal index

		A[i, i+1] =  dx[i-1]/dx[i]/(dx[i] + dx[i-1]) #1 above diagonal
		A[i, i]   = (dx[i]-dx[i-1])/(dx[i]*dx[i-1])  #diagonal
		A[i, i-1] = -dx[i]/dx[i-1]/(dx[i] + dx[i-1]) #1 below diagonal
	
		Abulk[1:-1,:] = A[1:-1,:]
		return Abulk

	def matrix_laplacian(self):
		dx= self.dx
		x = self.xs
		i = np.arange(self.Nx)[1:-1] # tri-diagonal index

		# 2/r φ'
		A1 = np.zeros((self.Nx, self.Nx)) # Is N x N
		A1[i, i+1] = 2/x[i] *  dx[i-1]/dx[i]/(dx[i] + dx[i-1]) #1 above diagonal
		A1[i, i]   = 2/x[i] * (dx[i]-dx[i-1])/(dx[i]*dx[i-1])  #diagonal
		A1[i, i-1] = 2/x[i] * -dx[i]/dx[i-1]/(dx[i] + dx[i-1]) #1 below diagonal
		
		# φ"		
		A2 = self.matrix_d2fdx2()

		A = A1 + A2 
		Abulk =  np.zeros((self.Nx, self.Nx)) # Is N x N
		Abulk[1:-1,:] = A[1:-1,:]


		return Abulk