import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

class HNC:
    def __init__(self, Gamma=50, kappa=2.0, alpha=1.0, potential_func=None, num_iterations=1000, R_max=25.0, N_bins=512):
        self.Gamma = Gamma
        self.kappa = kappa
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.R_max = R_max
        self.N_bins = N_bins
        
        self.del_r = R_max / N_bins
        self.r_array = np.linspace(self.del_r / 2, R_max - self.del_r / 2, N_bins)
        self.energy_iter = np.zeros(num_iterations)
        self.pressure_iter = np.zeros(num_iterations)
        
        self.del_k = np.pi / ((N_bins + 1) * self.del_r)
        self.K_max = self.del_k * N_bins
        self.k_array = np.linspace(self.del_k / 2, self.K_max - self.del_k / 2, N_bins)

        self.fact_r_2_k = 2 * np.pi * self.del_r
        self.fact_k_2_r = self.del_k / (4. * np.pi**2)
        self.dimless_dens = 3. / (4 * np.pi)
        
        if potential_func is None:
            self.potential_func = self.yukawa_potential
        else:
            self.potential_func = potential_func

    def yukawa_potential(self, r_in):
        return [self.Gamma * np.exp(-self.kappa * r_in) / r_in, self.Gamma * np.exp(-(self.alpha + self.kappa) * r_in) / r_in]

    def FT_r_2_k(self, input_array):
        from_dst = self.fact_r_2_k * fftpack.dst(self.r_array * input_array)
        return from_dst / self.k_array

    def FT_k_2_r(self, input_array):
        from_idst = self.fact_k_2_r * fftpack.idst(self.k_array * input_array)
        return from_idst / self.r_array

    def initial_c_k(self, k_in):
        return -4 * np.pi * self.Gamma / (self.k_array**2 + self.kappa**2)

    def u_long_l(self, k_in):
        return 4 * np.pi * self.Gamma * (self.alpha**2 + 2 * self.alpha * self.kappa) / ((self.k_array**2 + self.kappa**2) * (self.k_array**2 + (self.alpha + self.kappa)**2))

    def HNC_solve(self):
        u_r, u_s_r = self.potential_func(self.r_array)
        c_k = self.initial_c_k(self.k_array)
        u_l_k = self.u_long_l(self.k_array)
        c_s_k = c_k + u_l_k

        for iteration in np.arange(self.num_iterations):
            gamma_s_k = (self.dimless_dens * c_s_k * c_k - u_l_k) / (1 - self.dimless_dens * c_k)
            gamma_s_r = self.FT_k_2_r(gamma_s_k)
            g_r = np.exp(gamma_s_r - u_s_r)
            self.energy_iter[iteration] = 1.5 * self.del_r * np.sum(self.r_array**2 * u_r * (g_r - 1))
            d_u_d_r = -self.Gamma * np.exp(-self.kappa * self.r_array) * (1 + self.kappa * self.r_array) / self.r_array**2
            self.pressure_iter[iteration] = -0.5 * self.del_r * np.sum(self.r_array**3 * d_u_d_r * (g_r - 1))
            new_c_s_r = g_r - 1 - gamma_s_r
            c_s_k = self.FT_r_2_k(new_c_s_r)
            c_k = c_s_k - u_l_k

        self.g_r = g_r
        self.c_k = c_k
        self.u_l_k = u_l_k
        self.c_s_k = c_s_k
        self.S_k = 1+self.dimless_dens*self.c_k/(1-self.dimless_dens*self.c_k)

    def plot_hnc(self):
        # plt.figure(figsize=(35,20), dpi= 80, facecolor='g', edgecolor='r')
        plt.figure(figsize=(35,20))

        plt.subplot(3,1,1)
        plt.plot(self.r_array,self.r_array*np.log(self.g_r), label='from g(r)')
        # plt.plot(r_array,-Gamma*np.exp(-np.sqrt(kappa**2 + 3*Gamma/(1 + 3*Gamma))*r_array), label='exp form')
        plt.ylabel('$g(r)$')
        plt.xlabel('$r$')
        plt.xlim(0,6)
        plt.legend()
        # plt.grid()

        plt.subplot(3,1,2)
        plt.plot(self.k_array,self.S_k)
        plt.ylabel('$S(k)$')
        plt.xlabel('$k$')
        plt.xlim(0,12)
        # plt.grid()

        plt.subplot(3,1,3)
        plt.plot(self.k_array,self.FT_k_2_r(self.c_k))
        plt.ylabel('$c(r)$')
        plt.xlabel('$r$')
        plt.xlim(0,20)
        # plt.grid()

        plt.show()
        # plt.tight_layout()

if __name__=='__main__':
    hnc = HNC()
    hnc.HNC_solve()
    hnc.plot_hnc()

