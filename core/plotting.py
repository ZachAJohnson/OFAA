import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import colors
from hnc.hnc.constants import *
from pandas import read_csv
from .config import CORE_DIR, PACKAGE_DIR

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


def compare_aa(aa_list, axs=None, name='', make_petrov_comparison = True):
    if make_petrov_comparison:
        petrov = read_csv("../data/George_Petrov/GP_TFDW_Al_1eV_solid.dat",header=0, comment='#', delim_whitespace=True)

    eV = 0.0367512 # So 4 eV = 4 * 0.036.. in natural units
    if axs is None:
        fig, axs = plt.subplots(ncols=2,figsize=(20,8),facecolor='w')

    # Density * 4pi r^2 plot
    for ax in axs:
        for aa, color in zip( aa_list, color_cycle):
            rs = aa.rs
            factor = 4*np.pi*aa.grid.xs**2
            if make_petrov_comparison:
                ax.plot(np.array(petrov['r/R'])*aa.rs, 4*π*(np.array(petrov['r/R'])*aa.rs)**2*(petrov['ne']/m_to_AU**3), 'k--', label="Petrov AA")
            ax.plot(aa.grid.xs, aa.ne*factor ,'-',color=color, label=r'$n_e$: '+ name)
            ax.plot(aa.grid.xs, aa.nb*factor,'--',color=color,  label=r'$n_b$: ' + name)
            ax.plot(aa.grid.xs, aa.nf*factor,':',color=color,  label=r'$n_f$: ' + name)
            # ax.plot(aa.grid.xs, aa.δn_f*factor,'-',color='b',  label=r'$n^{scr} = n^{PA} - n^{ion}$: ' + name)
            # ax.plot(aa.grid.xs, (aa.δn_f+aa.nb)*factor,'--',color='b',  label=r'$n^{PA}=n-n^{empty}$: ' + name)
            # ax.plot(aa.grid.xs, (aa.ρi - aa.empty_ne )*factor,':',color='b',  label=r'$\rho^i - n^{empty}$: ' + name)
            if aa.rs != aa.R:
                ax.plot(aa.grid.xs, aa.ρi*factor,'-',color='r',  label=r'$\rho_i$: ' + name)
                ax.plot(aa.grid.xs, (aa.ρi-aa.ne)*factor,'--',color='r',  label=r'$\rho_i+\rho_e$: ' + name)
        # axs[0].plot(aa.grid.xs, aa.δn_f*factor,'--',color='g',  label=r'$n_e^{sc}$: ' + name)


    axs[0].set_ylabel(r'$4 \pi r^2 n_e(r) $ [A.U.]',fontsize=20)
    axs[0].set_ylim(-1e1, 1e3)
    axs[0].set_yscale('symlog',linthresh=1e-1)
    axs[0].set_xscale('log')
    axs[0].set_xlim(aa.grid.xs[0],aa.grid.xs[-1])

    axs[1].set_ylabel(r'$4 \pi r^2 n_e(r) $ [A.U.]',fontsize=20)
    axs[1].set_ylim(0, 13)
    axs[1].set_xlim(0, np.min([aa.R,1.3*aa.rs]))

        
    for ax in axs:
        ax.set_xlabel(r'$|r-R_1|$ [A.U.]',fontsize=20)
        ax.legend(loc="upper right",fontsize=20,labelspacing = 0.1)
        ax.tick_params(labelsize=20)
        ax.grid(which='both',alpha=0.4)

        # make textbox
        text = ("{0}\n".format(aa.name)+ 
            r"$r_s$ = " + "{0},    ".format(np.round(aa.rs,2)) +
            r"$R_{NPA}$ = " + "{0}\n".format(aa.R)  +
                r"$T_e$ = " + "{0} [A.U.] = {1} eV\n".format(np.round(aa.Te,2),np.round(aa.Te/eV,2)) + r"$\mu$ = " + "{0} [A.U.]\n".format(np.round(aa.μ,2)) +
                r"$Z^\ast = $" + "{0}".format(np.round(aa.Zstar,2))  )

        props = dict(boxstyle='round', facecolor='w')
        ax.text(0.25,0.95, text, fontsize=15, transform=ax.transAxes, verticalalignment='top', bbox=props)

    plt.tight_layout()
    name = "AA_densities_{0}_rs{1}_{2}eV_R{3}.png".format(aa.name, np.round(aa.rs,2), np.round(aa.Te/eV,2) ,np.round(aa.R))
    plt.savefig(os.path.join(PACKAGE_DIR,"media",name), dpi=300, bbox_inches='tight',facecolor="w")
    
    return axs

def plot_convergence(aa, axs=None):
    if axs is None:
        fig, axs = plt.subplots(ncols=3,figsize=(18,5),facecolor='w', dpi=200)
    
    # ELectric potential
    slice_by_num = 10
    ax = axs[0]
    colors = plt.cm.coolwarm(np.linspace(0, 1,len(aa.φe_list[::slice_by_num])))
    for i, (φe, ne, μ, ne_bar) in enumerate(zip(aa.φe_list[::slice_by_num], aa.ne_list[::slice_by_num], aa.μ_list[::slice_by_num],aa.ne_bar_list[::slice_by_num])):
        if i ==0 or i==len(aa.φe_list[::slice_by_num])-1:
            ax.plot(aa.grid.xs, (φe+aa.φion),linewidth=1,color=colors[i],alpha=1, label=r'$\phi_e$'.format(i))
            # ax.plot(aa.grid.xs, -aa.grid.dfdx(φe+aa.φion),linewidth=1,color=colors[i],alpha=1, label=r'$\vec E$'.format(i))
            # ax.plot(aa.grid.xs, aa.get_βVeff(φe, ne, ne_bar),linewidth=1,color=colors[i],alpha=1, label=r'$\beta V_{{\rm eff}}$'.format(i))
        else:
            ax.plot(aa.grid.xs, φe+aa.φion,linewidth=1,color=colors[i],alpha=1)
            # ax.plot(aa.grid.xs, -aa.grid.dfdx(φe+aa.φion),linewidth=1,color=colors[i],alpha=1)
            # ax.plot(aa.grid.xs, aa.get_βVeff(φe, ne, ne_bar),linewidth=1,color=colors[i],alpha=1)
    # ax.plot(aa.grid.xs, aa.get_φe( (aa.ρi - aa.ne) )[0] + aa.φion  ,'k:', label=r'$\phi$ check') 
    ax.set_yscale('symlog',linthresh=1e-10)
    ax.plot(aa.grid.xs, aa.φe_init + aa.φion, 'k:')

    # number density
    ax = axs[1]
    for i, (ne, ρi) in enumerate(zip(aa.ne_list[::slice_by_num], aa.ρi_list[::slice_by_num])):
        if i ==0 or i==len(aa.ne_list[::slice_by_num])-1:
            if i==0: 
                ne_bar_0 = ne[-1]
            ax.plot(aa.grid.xs, [aa.grid.integrate_f(-ne + ρi, end_index = index) + aa.Z for index in range(len(aa.grid.xs))],linewidth=1,color=colors[i],alpha=1, label=r'$Q(r)$'.format(i*slice_by_num))
            # ax.plot(aa.grid.xs, -ne + 0*ρi + ne[-1],linewidth=1,color=colors[i],alpha=1, label=r'$\rho$'.format(i))
            # ax.plot(aa.grid.xs, -ne + ρi,linewidth=1,color=colors[i],alpha=1, label=r'$\rho$'.format(i))
            # ax.plot(aa.grid.xs, -ne + ne_bar_0,linewidth=1,color=colors[i],alpha=1, label=r'$\rho$'.format(i))
            # ax.plot(aa.grid.xs, -ρi[-1] + ρi,linewidth=1,color=colors[i],alpha=1, label=r'$\rho$'.format(i))
            # pass
        else:
            ax.plot(aa.grid.xs, [aa.grid.integrate_f(-ne + ρi, end_index = index) + aa.Z for index in range(len(aa.grid.xs))],linewidth=1,color=colors[i],alpha=1)
            # print(ne[-1]-ne_bar_0)
            # ax.plot(aa.grid.xs, -ne + ρi ,linewidth=1,linestyle='-',color=colors[i],alpha=0.5)
            # ax.plot(aa.grid.xs, -ne + ne_bar_0 ,linewidth=1,linestyle='-',color=colors[i],alpha=0.5)
            # ax.plot(aa.grid.xs, -ρi[-1]+ ρi ,linewidth=1,linestyle='-',color=colors[i],alpha=0.5)
            pass
    # for i in range(int(aa.R/aa.rs)):
    #     ax.axvline(aa.rs*i, color='k', linestyle='--', alpha=0.2)
    # ax.set_xscale('log')
    # ax.plot(aa.grid.xs, (aa.gii)-1, 'k:')
    ax.set_yscale('symlog',linthresh=1e-4)

            
    # number density
    ax = axs[2]
    for i, (φe, ne, μ, ne_bar) in enumerate(zip(aa.φe_list[::slice_by_num], aa.ne_list[::slice_by_num], aa.μ_list[::slice_by_num],aa.ne_bar_list[::slice_by_num])):
        ne_TF = aa.get_ne_TF(φe, ne, μ, ne_bar)
        if i ==0 or i==len(aa.ne_list[::slice_by_num])-1:
            ax.plot(aa.grid.xs, ne - ne_TF,linewidth=1,color=colors[i],alpha=1, label=r'$n_e-n_e^{{TF}}$, iter: {0}'.format(i))
            ax.plot(aa.grid.xs, ne_TF-0*ne_TF[-1],linewidth=1,color=colors[i],linestyle=':',alpha=1, label=r'$n_e^{{TF}} - n_e^{{TF}}[-1]$, iter: {0}'.format(i*slice_by_num))
            ax.plot(aa.grid.xs, ne-0*ne[-1],linewidth=1,color=colors[i],linestyle='--',alpha=1, label=r'$n_e - n_e[-1]$, iter: {0}'.format(i))
            # pass
        else:
            ax.plot(aa.grid.xs, ne - ne_TF, linewidth=1,linestyle='-',color=colors[i],alpha=0.5)
            ax.plot(aa.grid.xs, ne_TF-0*ne_TF[-1],linewidth=1,linestyle=':',color=colors[i],alpha=1)
            ax.plot(aa.grid.xs, ne-0*ne[-1], linewidth=1,linestyle='--',color=colors[i],alpha=0.5)
            

            # print(np.abs(ne/ne_TF-1))
            # print(np.where( np.abs(ne/ne_TF-1)>1e-6 ))
            # print(aa.grid.xs[np.where( np.abs(ne/ne_TF-1) > 1e-8)][0]/10, 1/(np.sum(aa.grid.xs*np.abs(ne/ne_TF-1))/np.sum(np.abs(ne/ne_TF-1))))
    # ax.set_ylim()
    # ax.plot(aa.grid.xs, npa.get_ne_TF(npa.φe, npa.ne, npa.μ, npa.ne_bar) - npa.get_ne_TF(npa.φe, npa.ne, npa.μ, npa.ne_bar)[-1] , 'k:')
    ax.plot(aa.grid.xs, aa.get_ne_TF(aa.φe, aa.ne_bar*np.ones_like(aa.ne), aa.μ, aa.ne_bar) - aa.get_ne_TF(aa.φe, aa.ne_bar*np.ones_like(aa.ne), aa.μ, aa.ne_bar)[-1] , 'k:')
    ax.set_yscale('symlog',linthresh=1e-10)
    # ax.set_xscale('log')
    for ax in axs:
        ax.legend(fontsize=14)
        ax.tick_params(labelsize=14)
        ax.plot(aa.grid.xs, np.zeros_like(aa.ne),'k', alpha=0.2)
        ax.set_xlim(0,1.5)
    plt.tight_layout()
    return fig, axs
    
def plot_Uei(aa_list, axs=None, name='', make_petrov_comparison = True):
    # eV = 0.0367512 # So 4 eV = 4 * 0.036.. in natural units
    if axs is None:
        fig, axs = plt.subplots(1,2, figsize=(10,4))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for aa, color in zip(aa_list, colors):
        axs[0].plot(aa.iet.k_array/aa.rs, aa.Uei_iet_k*(aa.iet.k_array/aa.rs)**2/(4*π) ,'--.',color=color, label=aa.aa_type ) # Need to muultiply by some rs power???
        axs[0].plot(aa.iet.k_array/aa.rs, -aa.Zstar*np.ones_like(aa.iet.k_array),':', color=color, label=f"Coulomb, Z={aa.Zstar:0.3f}")

    # axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_xlabel(r"$k$ [au]")
    axs[0].set_ylabel(r"$U_{ei}$ [au]")

    for aa, color in zip(aa_list, colors):
        axs[1].plot(aa.iet.k_array/aa.rs, aa.uii_k_eff_iet,'-', color=color, label=aa.aa_type)
        axs[1].plot(aa.iet.k_array/aa.rs, 4*π*aa.Zstar**2/(aa.iet.k_array/aa.rs)**2,'--', color=color, label=r"{0}: $u_{{ii}}^0$, Z={1:0.3f}".format(aa.aa_type, aa.Zstar))
        axs[1].plot(aa.iet.k_array/aa.rs, 1/(aa.iet.k_array**2/aa.rs**2/(4*π*aa.Zstar**2) + 1/aa.uii_k_eff_iet[0] ),'--', color=color, label=r"{0}: $u_{{ii}}^0$, Z={1:0.3f}".format(aa.aa_type, aa.Zstar))


    axs[1].set_yscale('symlog',linthresh=1e-4)
    axs[1].set_ylim(0,1e5)
    axs[1].set_xscale('log')
    axs[1].set_xlabel(r"$k$ [au]")
    axs[1].set_ylabel(r"$u^{\rm eff}_{ii}$ [au]")


    for ax in axs:
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3, which='both')
    plt.tight_layout()

    return fig, axs

def plot_Uii(aa_list, axs=None, name='', make_petrov_comparison = True):
    # eV = 0.0367512 # So 4 eV = 4 * 0.036.. in natural units
    if axs is None:
        fig, axs = plt.subplots(1,2, figsize=(10,4))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for aa, color in zip(aa_list, colors):
        axs[0].plot(aa.iet.r_array*aa.rs, aa.Zstar/(aa.iet.r_array*aa.rs)+0*aa.Zstar/aa.R, color=color, label=f"Coulomb, Z={aa.Zstar:0.3f}")
        axs[0].plot(aa.iet.r_array*aa.rs, np.abs(aa.Uei_iet) + 0*aa.Zstar/aa.R ,':',color=color, label=aa.aa_type ) # Need to muultiply by some rs power???
        
    axs[0].axvline(aa.rs, color='k')
    
    axs[0].set_yscale('log')#,linthresh=1e-1)
    axs[0].set_xscale('log')
    axs[0].set_xlabel(r"$r$ [au]")
    axs[0].set_ylabel(r"$|U_{ei}|$ [au]")

    for aa, color in zip(aa_list, colors):
        axs[1].plot(aa.iet.r_array*aa.rs, 1/aa.Te * aa.uii_r_eff_iet, color=color,linestyle='-', label=aa.aa_type)
        axs[1].plot(aa.iet.r_array*aa.rs, 1/aa.Te *aa.Zstar**2/(aa.iet.r_array*aa.rs ), color=color ,linestyle='--', label=f'Coulomb Z={aa.Zstar:0.3f}')
        

    axs[1].set_yscale('symlog',linthresh=1e-4)
    axs[1].set_xscale('log')
    axs[1].set_ylabel(r"$\beta U_{ii}$ [au]")
    axs[1].set_xlabel(r"$r$ [au]")
    axs[1].set_xlim(1,None)


    for ax in axs:
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    return fig, axs


def plot_hii(aa_list, axs=None, name='', make_petrov_comparison = True):
    # eV = 0.0367512 # So 4 eV = 4 * 0.036.. in natural units
    if axs is None:
            fig, axs  = plt.subplots(1,2, figsize=(8,4))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plusone_list = [1, 0]
    for ax, plusone in zip(axs, plusone_list):
        for aa, color in zip(aa_list, colors):    
            ax.plot(aa.iet.r_array, aa.iet.h_r_matrix[0,0] + plusone, label=r"{0}: $u_{{ii}}^{{\rm eff}}$".format(aa.aa_type),linewidth=1, zorder=5)

    ax = axs[0]
    ax.set_xlabel(r"$r/r_i$")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0.7,2.5)
    ax.set_ylim(0,2.5)
    ax.legend(fontsize=10)

    ax = axs[1]
    ax.set_xlabel(r"$r/r_i$")
    ax.set_ylabel("h(r)")
    ax.set_xlim(0.7,10)
    ax.set_ylim(0,2.5)
    ax.set_yscale('symlog', linthresh=1e-4)

    ax.legend(fontsize=8, loc='upper right')
    
    for ax in axs:
        ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    return fig, axs
