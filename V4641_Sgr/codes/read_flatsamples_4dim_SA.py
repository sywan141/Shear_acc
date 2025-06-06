import numpy as np
from scipy.stats import chi2, gaussian_kde
import astropy.units as u
from multiprocessing import Pool
from NAIMA_RAD import Elect_spec, NAIMA_IC, NAIMA_SYN, IC_SA
from functools import partial
from scipy.integrate import simps
import corner
import matplotlib.pyplot as plt
import astropy.constants as const

datadir = '/home/wsy/V4641_Sgr/results/paras/'
tar_dir = '/home/wsy/V4641_Sgr/SA_corners/'
DOF = 17-4
R= [5.0,1.0,0.5]#[5.0, 1.0, 0.5]
ndim = 4
spectrum_energy_ic = np.logspace(-3.1,3.2,1000)
c = (const.c.cgs).value

def process_paras(paras, R):
    lgB, rate, lgK, lgrho = paras
    sed_IC = IC_SA(R,lgB,rate,lgK,lgrho)[0]
    return sed_IC

for i in R:
    flat_samples = np.loadtxt(datadir + 'SA_4dim_model_xi=1_%spc_limited_DIFFUSION_MODIFIED_1_GalacticExtinction_q=2_widerange/'%i+
    'paras_lgB0=-0.6_rate=0.040961672771081724_lgK=52.22_n=60000_weighted.txt')
    chi2_samples = np.loadtxt(datadir + 'SA_4dim_model_xi=1_%spc_limited_DIFFUSION_MODIFIED_1_GalacticExtinction_q=2_widerange/'%i+
    'chi2_lgB0=-0.6_rate=0.040961672771081724_lgK=52.22_n=60000_weighted.txt')

    # 在双峰/有偏分布的情况下，额外选取在一个sigma内的所有样本没有任何意义。最佳拟合值可能为两个峰中的任意一个，因此这里不需要percentile函数, 直接给出在chi2分布空间上的允许值
    
    mask1 = np.argmin(chi2_samples)
    chi2_min = np.min(chi2_samples)
    theta_best = flat_samples[mask1]

    delt_chi2 = chi2_samples-chi2_min
    #print(np.max(delt_chi2))
    delt_crit = chi2.ppf(0.68, DOF)
    mask2 = (delt_chi2<=delt_crit)
    chi2_subset_final = chi2_samples[mask2]
    paras_subset_final = flat_samples[mask2]
    length = len(paras_subset_final)

    lgB_st = paras_subset_final[:,0]
    eta_st = paras_subset_final[:,1]
    lgK_st = paras_subset_final[:,2]
    lgrho_st = paras_subset_final[:,3]
    #print('(%s, %s,%s,%s,%s,%s,%s,%s)'%(np.min(lgB_st),np.max(lgB_st), np.min(eta_st),np.max(eta_st),np.min(lgK_st),np.max(lgK_st),np.min(lgrho_st),np.max(lgrho_st)) )
   
    data = np.column_stack((lgB_st, eta_st, lgK_st, lgrho_st))
    
    labels = [r"$\lg B$",r"$\eta$",'lgK',r"$\lg \rho$"]
    fig = corner.corner(
        data,
        labels=labels,
        show_titles=True,  
        title_kwargs={"fontsize": 12}, 
        bins = 30,         
        smooth=1.0,
        hist_kwargs={"density": True}        
    )
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for j in range(ndim):
        ax = axes[j, j]
        data_j = data[:, j]
        kde = gaussian_kde(data_j)
        x = np.linspace(data_j.min(), data_j.max(), 1000)
        y = kde(x)
        ax.plot(x, y, color='deepskyblue', lw=1.5, ls='--', label='KDE')
    corner.overplot_lines(fig, theta_best, color="C1") # parametres when chi2 reaches minimum
    corner.overplot_points(fig, theta_best[None], marker="s", color="C1")
    plt.savefig(tar_dir+'SA_corners_%s_chi2.png'%i)
    plt.savefig(tar_dir+'SA_corners_%s_chi2.pdf'%i,format="pdf", dpi=300, bbox_inches="tight")

    length = len(paras_subset_final)
    mask_cut = np.linspace(0,length-2,num=100).astype(int)

    # record ic radiation
    lgB_best,rate_best,lgK_best,lgrho_best = theta_best
    beta_Alf = (((10**lgB_best)*(u.uG)).to(u.G)).value/(c*np.sqrt(4*np.pi*10**lgrho_best))
    Gamma_Alf = 1/(np.sqrt(1-beta_Alf**2))
    print(rate_best**2/(beta_Alf*Gamma_Alf**2)**2)
    R_jet = ((i*(u.pc)).cgs).value
    #SYN_all = []
    IC_all = []
    with Pool(processes=100) as pool:
        results = pool.map(partial(process_paras, R=i), paras_subset_final[mask_cut]) 
    #SYN_all = [r[0] for r in results]
    IC_all = [r for r in results]
    #print(IC_all)
    IC_tot, IC_cmb, IC_fir, IC_nir= IC_SA(i,lgB_best,rate_best,lgK_best,lgrho_best)
    sed_min = np.min(IC_all,axis=0)
    sed_max = np.max(IC_all,axis=0)
    filei = np.column_stack((spectrum_energy_ic,IC_tot, IC_cmb, IC_fir, IC_nir, sed_min, sed_max))
    np.savetxt(tar_dir+'IC_%spc_spec_SA.txt'%i,filei)