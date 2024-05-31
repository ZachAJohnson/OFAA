import numpy as np
from scipy import fftpack
from scipy.interpolate import interp1d
from pandas import read_csv as read_csv

import matplotlib.pyplot as plt


def make_potential_func(r_data, potential_data):
    # r_data = self.lin.r_list/self.rs
    # potential_data = 1/self.T*self.lin.vii
    # potential_data = 1/self.T * np.array([self.Zstar**2/r * np.exp(-self.lin.kTF*r) for r in self.lin.r_list])


    filter_cutoff = 0.1
    long_range_potential, short_range_potential = filter_potential_data(r_data, potential_data, filter_cutoff)

    # Create interpolation functions for the total, long-range, and short-range potentials
    total_potential_interp       = interp1d(r_data, potential_data,        kind='linear', bounds_error=False, fill_value='extrapolate')
    long_range_potential_interp  = interp1d(r_data, long_range_potential,  kind='linear', bounds_error=False, fill_value='extrapolate')
    short_range_potential_interp = interp1d(r_data, short_range_potential, kind='linear', bounds_error=False, fill_value='extrapolate')

    def HNC_potential( r_in):
        short_range_potential = short_range_potential_interp(r_in)
        long_range_potential  = total_potential_interp(r_in) - short_range_potential #long_range_potential_interp(r_in)

        return [ short_range_potential, long_range_potential]

    return HNC_potential
# def make_potential_func(r_data, potential_data):
#     # r_data = self.lin.r_list/self.rs
#     # potential_data = 1/self.T*self.lin.vii
#     # potential_data = 1/self.T * np.array([self.Zstar**2/r * np.exp(-self.lin.kTF*r) for r in self.lin.r_list])


#     filter_cutoff = 10
#     long_range_potential, short_range_potential = filter_potential_data(r_data, potential_data, filter_cutoff)

#     # Create interpolation functions for the total, long-range, and short-range potentials
#     total_potential_interp       = interp1d(r_data, potential_data,        kind='linear', bounds_error=False, fill_value='extrapolate')
#     long_range_potential_interp  = interp1d(r_data, long_range_potential,  kind='linear', bounds_error=False, fill_value='extrapolate')
#     short_range_potential_interp = interp1d(r_data, short_range_potential, kind='linear', bounds_error=False, fill_value='extrapolate')

#     def HNC_potential( r_in):
#         short_range_potential = 2e2*r_in**-6#total_potential_interp(r_in)
#         long_range_potential  = total_potential_interp(r_in) - 2e2*r_in**-6  #0*r_in#long_range_potential_interp(r_in)

#         return [ short_range_potential, long_range_potential]

#     return HNC_potential

def filter_potential_data(r_array, potential_data, filter_cutoff):
    # Calculate the Fourier transform of the potential data
    potential_fft = fftpack.fft(potential_data*r_array**2)

    # Define a frequency array based on the r_array
    freq = fftpack.fftfreq(len(r_array), r_array[1] - r_array[0])

    # Apply a low-pass filter for the long-range component
    long_range_fft = potential_fft * (np.abs(freq) < filter_cutoff)
    long_range_potential = np.real(fftpack.ifft(long_range_fft))

    # Apply a high-pass filter for the short-range component
    short_range_fft = potential_fft * (np.abs(freq) >= filter_cutoff)
    short_range_potential = np.real(fftpack.ifft(short_range_fft))

    return long_range_potential/r_array**2, short_range_potential/r_array**2


class HNC():
    def __init__(self, Gamma=5, kappa = 5.0, alpha=1.0, potential_func=None, tol=1e-5, num_iterations=10000, R_max=25.0, N_bins=512):
        self.Gamma = Gamma
        self.kappa = kappa
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.tol = tol
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
        u_s_r = self.Gamma * np.exp(-(self.alpha + self.kappa) * r_in) / r_in
        u_l_r = self.Gamma * ( np.exp(-self.kappa * r_in) - np.exp(-(self.alpha + self.kappa) * r_in) )/ r_in
        return [u_s_r, u_l_r]

    def FT_r_2_k(self, input_array):
        from_dst = self.fact_r_2_k * fftpack.dst(self.r_array * input_array)
        return from_dst / self.k_array

    def FT_k_2_r(self, input_array):
        from_idst = self.fact_k_2_r * fftpack.idst(self.k_array * input_array)
        return from_idst / self.r_array

    def initial_c_k(self, k_in):
        return -4 * np.pi * self.Gamma / (self.k_array**2 + self.kappa**2)

    # def u_long_l(self, k_in):
    #     return 4 * np.pi * self.Gamma * (self.alpha**2 + 2 * self.alpha * self.kappa) / ((self.k_array**2 + self.kappa**2) * (self.k_array**2 + (self.alpha + self.kappa)**2))

    def HNC_solve(self):
        u_s_r, u_l_r = self.potential_func(self.r_array)
        u_r = u_s_r + u_l_r
        c_k = self.initial_c_k(self.k_array)
        u_l_k = self.FT_r_2_k(u_l_r)
        c_s_k = c_k + u_l_k
        old_g_r = np.ones(self.N_bins)

        n_iter, err = 0, 1000
        while err > self.tol and n_iter < self.num_iterations:
            gamma_s_k = (self.dimless_dens * c_s_k * c_k - u_l_k) / (1 - self.dimless_dens * c_k)
            gamma_s_r = self.FT_k_2_r(gamma_s_k)
            g_r = np.exp(gamma_s_r - u_s_r)
            new_c_s_r = g_r - 1 - gamma_s_r
            c_s_k = self.FT_r_2_k(new_c_s_r)
            c_k = c_s_k - u_l_k
            
            err = np.linalg.norm(g_r-old_g_r)
            # print("Err: {0:.3e}".format(err))
            old_g_r = g_r

            n_iter+=1
        print("HNC Converges in {0} iterations, with err: {1:.3e}".format(n_iter, err))
        # self.pressure = -0.5 * self.del_r * np.sum(self.r_array**3 * d_u_d_r * (g_r - 1))
        # self.energy = 1.5 * self.del_r * np.sum(self.r_array**2 * u_r * (g_r - 1))

        self.g_r = g_r
        self.g_func = interp1d(self.r_array, g_r, kind='cubic',bounds_error=False, fill_value=(0,1) )

        self.c_k = c_k
        self.u_l_k = u_l_k
        self.c_s_k = c_s_k
        self.S_k = 1+self.dimless_dens*self.c_k/(1-self.dimless_dens*self.c_k)


    def plot_potential(self):
        fig, ax = plt.subplots(figsize=(10,8))

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        u_s_r, u_l_r = self.potential_func(self.r_array)
        u_r = u_s_r + u_l_r

        ax.plot(self.r_array, u_l_r, '-.', color=colors[0], linewidth = 0.5, label='long (l)')
        ax.plot(self.r_array, u_s_r, '--', color=colors[0], linewidth = 0.5, label='short (s)')
        ax.plot(self.r_array, u_r,   '--.' , color=colors[0], label='total')
        # ax.plot(self.r_array, 2e2*self.r_array**-6,   '-' , color=colors[2], label='total')

        if self.potential_func != self.yukawa_potential:

            #Yukawa
            yu_s_r, yu_l_r = np.array([self.yukawa_potential(r) for r in self.r_array]).T
            yu_r = yu_s_r + yu_l_r


            ax.plot(self.r_array, yu_l_r, '-.' , color=colors[1], linewidth=0.5, label='Yukawa l')
            ax.plot(self.r_array, yu_s_r, '--' , color=colors[1], linewidth=0.5, label='Yukawa s')
            ax.plot(self.r_array, yu_r,   '-'  , color=colors[1], label='Yukawa total')

        ax.set_title(r"$\Gamma$ = " + "{0:.3f}".format(self.Gamma) + r" , $\kappa$ = " + "{0:.3f}".format(self.kappa), fontsize=20)
        ax.set_yscale('symlog',linthresh=1e-2)
        ax.set_xscale('log')
        ax.set_xlim(np.min(self.r_array), np.max(self.r_array)) 
        ax.legend(fontsize=15)
        ax.tick_params(labelsize=15)
        ax.set_xlabel(r'$r/r_s$', fontsize=15)
        ax.set_ylabel(r'$\beta V$', fontsize=15)
        plt.show()


    def plot_hnc(self, data_to_compare=None, data_names=None):
        """
        Args: 
            list data_to_compare: list of filenames with g(r) data
            data_names: list of names to label files in plot
        """
        # plt.figure(figsize=(35,20), dpi= 80, facecolor='g', edgecolor='r')
        fig = plt.figure(figsize=(16,12), facecolor='w')

        gs = fig.add_gridspec(3,2)
    
        ax01 = fig.add_subplot(gs[0,:])
        # ax01 = fig.add_subplot(gs[0,1])
        ax1  = fig.add_subplot(gs[1,:]) 
        ax2  = fig.add_subplot(gs[2,:])
        

        # ax00.plot(self.r_array,self.g_r)
        # ax00.set_ylabel('$g(r)$')
        # ax00.set_xlabel(r'$r/r_s$')
        # ax00.set_xlim(0,2)
        # ax00.grid()

        ax01.plot(self.r_array, self.g_r, label='ZAA')
        if data_to_compare==None:
            pass
        else:
            for file_name, label in zip(data_to_compare,data_names):
                r_datas, g_datas = np.array(read_csv(file_name)).T
                ax01.plot(r_datas, g_datas, label=label)
                ax01.legend(fontsize=15)
        ax01.set_title(r"$\Gamma$ = " + "{0:.3f}".format(self.Gamma) + r" , $\kappa$ = " + "{0:.3f}".format(self.kappa), fontsize=20)
        ax01.set_ylabel('$g(r)$')
        ax01.set_xlabel(r'$r/r_s$')
        ax01.set_xlim(0,self.R_max)
        ax01.grid()
        ax01.set_ylim(0,2)
        
        ax1.plot(self.k_array,self.S_k, '--.')
        ax1.set_ylabel('$S(k)$')
        ax1.set_xlabel('$k$')
        ax1.set_xlim(0, int(3*2*np.pi) )
        ax1.grid()

        
        ax2.plot(self.r_array,self.FT_k_2_r(self.c_k), '--.')
        ax2.set_ylabel('$c(r)$')
        ax2.set_xlabel(r'$r/r_s$')
        ax2.set_xlim(0,self.R_max)
        ax2.grid()

        plt.tight_layout()
        plt.show()
        

if __name__=='__main__':
    #Runs Yukawa Potential
    hnc = HNC()
    hnc.HNC_solve()
    hnc.plot_hnc()

